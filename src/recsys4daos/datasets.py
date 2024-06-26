import datetime as dt
import pandas as pd

def to_uir(
    dfv: pd.DataFrame,
    col_user: str,
    col_item: str,
    rating_col: str,
    timestamp_col: str = 'timestamp',
) -> pd.DataFrame:
    """ Convert the dataframe to user-item-rating format """
    return dfv[['voter', 'proposal', 'date']].rename(
        columns={
            'voter': col_user,
            'proposal': col_item,
            'date': timestamp_col,
        }
    ).astype({
        col_user: str,
        col_item: str,
    }).assign(**{rating_col: 1})

def to_microsoft(dfv: pd.DataFrame):
    return to_uir(dfv, 'userID', 'itemID', 'rating')

def to_lenskit(dfv: pd.DataFrame):
    return to_uir(dfv, 'user', 'item', 'rating')

def filter_window_size(df: pd.DataFrame, end: dt.datetime, ws: str, timestamp_col='timestamp'):
    offset = pd.tseries.frequencies.to_offset(ws)
    return df[df[timestamp_col] > (end - offset)]
