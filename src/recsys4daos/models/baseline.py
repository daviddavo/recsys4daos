import pandas as pd

from recommenders.datasets.pandas_df_utils import filter_by

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
