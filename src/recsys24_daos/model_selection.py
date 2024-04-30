from typing import Optional, Generator
from collections import namedtuple

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

Fold = namedtuple('Fold', ['train', 'test', 'end', 'open_proposals'])
def cvtt_open(
    dfv: pd.DataFrame,
    freq: str,
    dfp: pd.DataFrame,
    *,
    remove_not_in_train_col=None,
    normalize=True,
    inclusive: IntervalClosedType = "left",
    item_col = 'itemID',
    time_col = 'timestamp',
) -> Generator[Fold, None, None]:
    """
    The developed method is, basically, a Cross-Validation Through Time but filtering
    out closed proposals.
    
    - https://arxiv.org/abs/2205.05393
    """
    times = pd.date_range(
        dfv[time_col].min(), dfv[time_col].max(), freq=freq, normalize=normalize, inclusive=inclusive
    )
    for train_end in times:
        train, test = get_train_test_from_time(
            train_end, dfv, 'timestamp', remove_not_in_train_col=remove_not_in_train_col
        )
        all_props = np.union1d(train[item_col], test[item_col])

        open_proposals = np.intersect1d(all_props, current_proposals(dfp, train_end))
        test_filtered = test[test[item_col].isin(open_proposals)]

        yield Fold(train, test_filtered, train_end, np.array(open_proposals))
