from typing import Optional
from pathlib import Path

import hashlib
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer


def get_embeddings_from_cache(
    dfp: pd.DataFrame,
    model: SentenceTransformer, 
    embeddings_cache_path: Path,
) -> pd.DataFrame:
    """Calculates the embeddings and saves them in the embeddigns_cache_path

    Args:
        dfp (pd.DataFrame): The proposals dataframe
        model (str): The name of the model
        embeddings_cache (Path): The path of the cache folder

    Returns:
        pd.DataFrame: The embeddings dataframe
    """
    modelhash = hashlib.sha256(str(model).encode())
    embeddings_cache_path.mkdir(exist_ok=True, parents=True)
    embeddings_cache_file = (embeddings_cache_path / modelhash.hexdigest()).with_suffix('.pkl')
    
    dfp = dfp.set_index('id')
    if embeddings_cache_file.exists():
        prev_embeddings = pd.read_pickle(embeddings_cache_file)
        remaining_embeddings_idx = dfp.index.difference(prev_embeddings.index)
    else:
        prev_embeddings = pd.Series(dtype=str)
        remaining_embeddings_idx = dfp.index

    if not remaining_embeddings_idx.empty:
        print("Some embeddings need to be calculated")
        remaining = dfp.loc[remaining_embeddings_idx]
        title_description = remaining['title'] + '\n' + remaining['description']
        title_description = title_description.fillna("")

        new_embeddings = pd.Series(
            list(model.encode(title_description, show_progress_bar=True, normalize_embeddings=True)),
            index=title_description.index,
        )

        all_embeddings = pd.concat((prev_embeddings, new_embeddings))
        all_embeddings.to_pickle(embeddings_cache_file)
    else:
        # print("All embeddings are already calculated")
        all_embeddings = prev_embeddings
    
    return all_embeddings.loc[dfp.index]

class NLPModel:
    def __init__(self, 
        dfp: pd.DataFrame,
        embeddings_cache: str | Path,
        *,
        model_name: str ='all-mpnet-base-v2',
        show_progress_bar: bool = True,
    ):
        self.dfp = dfp
        self.transformer_model = SentenceTransformer(model_name)
        self.show_progress_bar = show_progress_bar
        self.embeddings_cache: Path = embeddings_cache
        self.embeddings = None

    def fit(self):
        self.embeddings = get_embeddings_from_cache(self.dfp, self.transformer_model, self.embeddings_cache)

class NLPSimilarity(NLPModel):
    def __init__(self,
        dfp: pd.DataFrame,
        embeddings_cache: str | Path,
        col_user: str = 'userID',
        col_item: str = 'itemID',
        **kwargs,
    ):
        super().__init__(dfp, embeddings_cache, **kwargs)
        # self.train = train.reset_index(drop=True)
        self.train = None
        self.col_user = col_user
        self.col_item = col_item

    def fit(self, train):
        """Just calculate the embedding of each user (average of their previously voted proposals)
        """
        super().fit()
        self.train = train
        votes_embeddings = self.embeddings.loc[self.train[self.col_item]]
        self.voter_embeddings = self.train.groupby(self.col_user)[self.col_item].apply(lambda x: votes_embeddings.loc[x].mean(axis=0))

    def recommend_k_items(
        self, to_users, top_k=5, remove_seen=True, recommend_from=None, min_score = 0.0,
    ):
        voter_embeddings = self.voter_embeddings.loc[to_users]
        np_voter_embeddings = np.stack(voter_embeddings.to_numpy())
    
        prop_embeddings = self.embeddings
        if recommend_from is not None:
            assert len(recommend_from) >= top_k, "top_k should not be greater than the number of proposals to recommend"
            prop_embeddings = self.embeddings.loc[recommend_from]

        tr_embeddings = np.stack(prop_embeddings.to_numpy())
    
        scores = np_voter_embeddings @ tr_embeddings.T

        if remove_seen:
            trainu = self.train[self.train[self.col_user].isin(voter_embeddings.index) & self.train[self.col_item].isin(prop_embeddings.index)]
            itemID2idx = pd.Series(data=np.arange(len(prop_embeddings)), index=prop_embeddings.index)
            voterID2idx = pd.Series(data=np.arange(len(voter_embeddings)), index=voter_embeddings.index)

            scores[voterID2idx[trainu[self.col_user]], itemID2idx[trainu[self.col_item]]] = -np.inf
            
        best = (-scores).argsort(axis=1)
        topk = best[:, :top_k]

        # create df with columns
        # userID, itemID, prediction
        uid = np.repeat(np.arange(np_voter_embeddings.shape[0]), top_k)
        iid = topk.flatten()

        # transform int to id
        df = pd.DataFrame({
            self.col_user: voter_embeddings.index[uid],
            self.col_item: prop_embeddings.index[iid].astype(str),
            # 'prediction': 1,
            'prediction': scores[uid, iid],
        })
        return df[df['prediction'] > min_score].reset_index(drop=True)
