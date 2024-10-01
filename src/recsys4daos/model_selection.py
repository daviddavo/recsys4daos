from typing import Optional, Generator
from collections import namedtuple
import sys

import datetime as dt
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from pandas._typing import IntervalClosedType
from tqdm.autonotebook import tqdm

DEFAULT_CHECKPOINT_EVERY = dt.timedelta(seconds=60)

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
    last_fold: dt.datetime | np.datetime64 | str = None,
    inclusive: IntervalClosedType = "left",
    col_item = 'itemID',
    col_time = 'timestamp',
) -> Generator[Fold, None, None]:
    """
    The developed method is, basically, a Cross-Validation Through Time but filtering
    out closed proposals.
    
    - https://arxiv.org/abs/2205.05393
    """
    last_fold = np.datetime64(last_fold) if last_fold else None
    times = pd.date_range(
        dfv[col_time].min(), dfv[col_time].max(), freq=freq, normalize=normalize, inclusive=inclusive
    )
    if last_fold is not None:
        idx = np.searchsorted(times, last_fold)

        if idx >= len(times):
            raise ValueError(f'The last_fold is too big, last date is {times[-1]}')
        elif not times[idx] == last_fold:
            raise ValueError(f'The last_fold should be in the folds, nearest dates: {times[idx-1]}, {times[min(idx, len(times)-1)]}')

    for train_end in times:
        train, test = get_train_test_from_time(
            train_end, dfv, col_time, remove_not_in_train_col=remove_not_in_train_col
        )
        all_props = np.union1d(train[col_item], test[col_item])

        open_proposals = np.intersect1d(all_props, current_proposals(dfp, train_end))
        test_filtered = test[test[col_item].isin(open_proposals)]

        yield Fold(train, test_filtered, train_end, np.array(open_proposals))
        if train_end == last_fold:
            break

def save_progress(data, fname, keys):
    fname.parent.mkdir(parents=True, exist_ok=True)
    
    with open(fname, 'wb') as f:
        pickle.dump({
            'data': data,
            'keys': keys,
        }, f)

def load_progress(fname: Path, keys):
    if not fname.exists():
        return {}
        
    with open(fname, 'rb') as f:
        p = pickle.load(f)
        assert list(p['keys']) == list(keys), f'Not the same keys! {p["keys"]} != {keys}'
        return p['data']

def explore_hparams(func, param_grid, fname, checkpoint_every=DEFAULT_CHECKPOINT_EVERY):
    if not param_grid:
        return {}

    keys = list(sorted(param_grid[0].keys()))
    results = load_progress(fname, keys)
    if results:
        print("Restored checkpoint from", fname, "with", len(results), "results")
    
    try:
        next_checkpoint = dt.datetime.now() + checkpoint_every
        for p in tqdm(param_grid):
            p = dict(sorted(p.items()))
            assert list(p.keys()) == keys, f'Changing keys in the hparams is not supported {p.keys()} != {keys}'
            k = tuple(p.values())
            if k in results:
                continue
            
            results[k] = func(**p)

            if dt.datetime.now() > next_checkpoint:
                next_checkpoint = dt.datetime.now() + checkpoint_every
                save_progress(results, fname, keys)
                print(f"[{dt.datetime.now().isoformat()}] Saving checkpoint at {fname}")
    except KeyboardInterrupt:
        print("Interrupt received, returning", file=sys.stderr)

    save_progress(results, fname, keys)

    # Convert the results to records format
    asked = { tuple(v for _,v in sorted(p.items())) for p in param_grid }
    assert len(asked) == len(param_grid)
    return [ dict(zip(keys,v)) | r for v,r in results.items() if v in asked ]
