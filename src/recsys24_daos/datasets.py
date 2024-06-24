import datetime as dt
import pandas as pd


def to_microsoft(dfv: pd.DataFrame):
    df = dfv[['voter', 'proposal', 'date']].rename(
        columns={
            'voter': 'userID',
            'proposal': 'itemID',
            'date': 'timestamp',
        }
    )
    df['userID'] = df['userID'].astype('str')
    df['itemID'] = df['itemID'].astype('str')
    df['rating'] = 1

    return df

def filter_window_size(df: pd.DataFrame, end: dt.datetime, ws: str, timestamp_col='timestamp'):
    offset = pd.tseries.frequencies.to_offset(ws)
    return df[df[timestamp_col] > (end - offset)]
