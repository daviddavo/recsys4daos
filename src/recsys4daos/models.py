import itertools as it

import numpy as np
import pandas as pd
from recommenders.datasets.pandas_df_utils import filter_by
from recommenders.models.deeprec.models.graphrec.lightgcn import LightGCN
from recommenders.utils.python_utils import get_top_k_scored_items

class OpenPop:
    def __init__(self, train: pd.DataFrame):
        # Sorting by date for replicability
        self._train = train.sort_values('timestamp', ascending=False)
        self._df = None

    def fit(self):
        # Does nothing
        ...

    def recommend_k_items(self, users, top_k: int = 5, *, recommend_from=None, remove_train: bool = True):
        if recommend_from is None or len(recommend_from) == 0:
            msg = "recommend_from can't be emtpy in OpenPop"
            raise ValueError(msg)

        bestVotes = self._train[self._train['itemID'].isin(recommend_from)]['itemID'].value_counts()
        df = bestVotes.to_frame('prediction').reset_index().merge(pd.Series(users, name='userID'), how='cross')

        if remove_train:
            df = filter_by(df, self._train, ['userID', 'itemID'])

        df = df.groupby('userID').head(top_k).reset_index(drop=True)

        return df[['itemID', 'userID', 'prediction']]


class LightGCNCustom(LightGCN):
    # Copied from LightGCN.fit but RETURNING the data and deleting unnecessary things
    def __init__(self, hparams, data, seed=None):
        super().__init__(hparams, data, seed=seed)
        self.epochs_done = 0

    def fit_epoch(self):
        """Fit the model on self.data.train."""
        loss, mf_loss, emb_loss = 0.0, 0.0, 0.0
        n_batch = self.data.train.shape[0] // self.batch_size + 1
        for _ in range(n_batch):
            users, pos_items, neg_items = self.data.train_loader(self.batch_size)
            _, batch_loss, batch_mf_loss, batch_emb_loss = self.sess.run(
                [self.opt, self.loss, self.mf_loss, self.emb_loss],
                feed_dict={
                    self.users: users,
                    self.pos_items: pos_items,
                    self.neg_items: neg_items,
                },
            )
            loss += batch_loss / n_batch
            mf_loss += batch_mf_loss / n_batch
            emb_loss += batch_emb_loss / n_batch

        if np.isnan(loss):
            msg = "loss is nan."
            raise ValueError(msg)

        self.epochs_done += 1

        return loss, mf_loss, emb_loss

    def recommend_k_items(
        self,
        test,
        top_k=10,
        *,
        sort_top_k=True,
        remove_seen=True,
        use_id=False,
        recommend_from=None,
    ):
        """
        Copy-pasted from LightGCN but adding the `recommend_from` argument
        """
        data = self.data
        if not use_id:
            user_ids = np.array([data.user2id[x] for x in test[data.col_user].unique()])
        else:
            user_ids = np.array(test[data.col_user].unique())

        test_scores = self.score(user_ids, remove_seen=remove_seen)

        # ========== START NEW BEHAVIOUR
        if recommend_from is not None:
            assert len(recommend_from) > 0, "Recommend from can't be empty"
            # from_idx = np.array([data.item2id[x] for x in set(recommend_from) if x in data.item2id])
            from_idx = np.array([data.item2id[x] for x in set(recommend_from)])
            assert from_idx.size >= 0, "from_idx is empty"
            msk = np.ones(test_scores.shape[1], bool)
            msk[from_idx] = False

            # Set the score of that proposal to zero for every user
            test_scores[:, msk] = -np.inf
        # ========== END NEW BEHAVIOUR

        top_items, top_scores = get_top_k_scored_items(scores=test_scores, top_k=top_k, sort_top_k=sort_top_k)

        df = pd.DataFrame(
            {
                data.col_user: np.repeat(test[data.col_user].drop_duplicates().values, top_items.shape[1]),
                data.col_item: top_items.flatten() if use_id else [data.id2item[item] for item in top_items.flatten()],
                data.col_prediction: top_scores.flatten(),
            }
        )

        return df.replace(-np.inf, np.nan).dropna()
