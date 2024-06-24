import os, sys
from pathlib import Path

import pandas as pd

DEFAULT_DATA_PATH = '../data'
DEFAULT_OUTPUT_PATH = '../data/outputs'

def _gen_fname(prefix, org_name, splits_freq, normalize, *, ext='csv', **kwargs) -> Path:
    other_args = "-".join([ f"{k}={v}" for k,v in kwargs.items() if v ])
    
    other_args = "-" + other_args if other_args else ""
    base_path = Path(os.getenv('RS4DAO_OUTPUT_PATH', DEFAULT_OUTPUT_PATH)).expanduser()
    return base_path / org_name / f'{prefix}_{splits_freq}{"_normalize" if normalize else ""}_{other_args}.{ext}'

def lightgcn_ray_tune_fname(org_name, splits_freq, normalize, optim_metric, fold='glob'):
    """ The ray tune file name """
    _fold = fold
    if _fold == 'glob':
        _fold = '[0-9]*'
    
    return f'{org_name}/LightGCN_{splits_freq}{"_normalize" if normalize else ""}_{optim_metric}_fold={_fold}'

def _save_pq(df: pd.DataFrame, prefix, *args, **kwargs):
    fname = _gen_fname(prefix, *args, ext='pq', **kwargs)
    fname.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(fname)

def _load_pq(prefix, *args, **kwargs):
    fname = _gen_fname(prefix, *args, ext='pq', **kwargs)
    return pd.read_parquet(fname)

def save_openpop(df: pd.DataFrame, org_name: str, splits_freq: str, splits_normalize: bool):
    _save_pq(df, 'baseline/openpop', org_name, splits_freq, splits_normalize)

def load_openpop(df: pd.DataFrame, org_name: str, splits_freq: str, splits_normalize: bool):
    return _load_pq(df, 'baseline/openpop', org_name, splits_freq, splits_normalize)

def save_perfect(df: pd.DataFrame, org_name: str, splits_freq: str, splits_normalize: bool):
    _save_pq(df, 'baseline/perfect', org_name, splits_freq, splits_normalize)

def load_perfect(df: pd.DataFrame, org_name: str, splits_freq: str, splits_normalize: bool):
    return _load_pq(df, 'baseline/perfect', org_name, splits_freq, splits_normalize)

def load_proposals(org_name, base=DEFAULT_DATA_PATH):
    base = Path(base).expanduser()
    df = pd.read_csv(base / org_name / 'proposals.csv', parse_dates=['date', 'start', 'end'])
    return df

def load_votes(org_name, base=DEFAULT_DATA_PATH):
    base = Path(base).expanduser()
    df = pd.read_csv(base / org_name / 'votes.csv', parse_dates=['date'])
    return df
