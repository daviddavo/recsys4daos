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
