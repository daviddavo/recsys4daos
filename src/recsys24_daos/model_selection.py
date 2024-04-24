from typing import Optional

import numpy as np
import pandas as pd
from pandas._typing import IntervalClosedType


def get_train_test_from_time(train_end_t, df, timestamp_col, remove_not_in_train_col: Optional[str] = None):
    train = df[df[timestamp_col] <= train_end_t]
    test = df[train_end_t < df[timestamp_col]]

    if remove_not_in_train_col is not None:
        msk = test[remove_not_in_train_col].isin(set(train[remove_not_in_train_col]))
        test = test[msk]

    return train, test


def current_proposals(dfp, t):
    """
    Open proposals: The ones that started before _t_, but are still open (close after _t_)
    """
    props = dfp[(dfp['start'] < t) & (t <= dfp['end'])]
    if 'id' in props.columns:
        return props['id']
    return props.index


def time_freq_split_current(
    dfv: pd.DataFrame,
    freq: str,
    dfp: pd.DataFrame,
    *,
    remove_not_in_train_col=None,
    normalize=True,
    inclusive: IntervalClosedType = "left",
):
    times = pd.date_range(
        dfv['timestamp'].min(), dfv['timestamp'].max(), freq=freq, normalize=normalize, inclusive=inclusive
    )
    for train_end in times:
        train, test = get_train_test_from_time(
            train_end, dfv, 'timestamp', remove_not_in_train_col=remove_not_in_train_col
        )
        all_props = np.union1d(train['itemID'], test['itemID'])

        open_proposals = np.intersect1d(all_props, current_proposals(dfp, train_end))
        test_filtered = test[test['itemID'].isin(open_proposals)]

        yield train, test_filtered, train_end, np.array(open_proposals)
