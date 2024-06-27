import os, sys
from pathlib import Path
import json

import pandas as pd

from recsys4daos.evaluation import all_metric_ks

DEFAULT_DATA_PATH = '../data'
DEFAULT_OUTPUT_PATH = '../data/outputs'
DEFAULT_CACHE_PATH = '../.cache'
MODEL_RESULTS_COLS = [
    'time_train',
    'time_rec',
    'time_eval',
]

def _gen_fname(prefix, org_name, splits_freq, normalize, *, ext='csv', **kwargs) -> Path:
    other_args = "-".join([ f"{k}={v}" for k,v in kwargs.items() if v ])
    
    fname = "_".join(filter(None, [prefix, splits_freq, "normalize" if normalize else None, other_args]))
    return f"{fname}.{ext}" 

def _gen_output_fname(prefix, org_name, splits_freq, normalize, *, ext='csv', **kwargs) -> Path:
    base_path = Path(os.getenv('RS4DAO_OUTPUT_PATH', DEFAULT_OUTPUT_PATH)).expanduser()
    return base_path / org_name / _gen_fname(prefix, org_name, splits_freq, normalize, ext=ext, **kwargs)

def _gen_cache_fname(prefix, org_name, splits_freq, normalize, *, ext='csv', **kwargs) -> Path:
    base_path = Path(os.getenv('RS4DAO_CACHE_PATH', DEFAULT_CACHE_PATH)).expanduser()
    return base_path / org_name / _gen_fname(prefix, org_name, splits_freq, normalize, ext=ext, **kwargs)

def lightgcn_ray_tune_fname(org_name, splits_freq, normalize, optim_metric, fold='glob'):
    """ The ray tune file name """
    _fold = fold
    if _fold == 'glob':
        _fold = '[0-9]*'
    
    return f'{org_name}/LightGCN_{splits_freq}{"_normalize" if normalize else ""}_{optim_metric}_fold={_fold}'

def hparams_progress(model_name, org_name, splits_freq, normalize, *, ext='pkl') -> Path:
    return _gen_cache_fname(f"hparams-{model_name}", org_name, splits_freq, normalize, ext=ext)

def _save_pq(df: pd.DataFrame, prefix, *args, **kwargs):
    fname = _gen_output_fname(prefix, *args, ext='pq', **kwargs)
    fname.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(fname)
    print(f"Saved dataframe into {fname}")

def _load_pq(prefix, *args, **kwargs):
    fname = _gen_output_fname(prefix, *args, ext='pq', **kwargs)
    return pd.read_parquet(fname)

def save_openpop(df: pd.DataFrame, org_name: str, splits_freq: str, splits_normalize: bool):
    _save_pq(df, 'baseline/openpop', org_name, splits_freq, splits_normalize)

def load_openpop(org_name: str, splits_freq: str, splits_normalize: bool):
    return _load_pq('baseline/openpop', org_name, splits_freq, splits_normalize)

def save_perfect(df: pd.DataFrame, org_name: str, splits_freq: str, splits_normalize: bool):
    _save_pq(df, 'baseline/perfect', org_name, splits_freq, splits_normalize)

def load_perfect(org_name: str, splits_freq: str, splits_normalize: bool):
    return _load_pq('baseline/perfect', org_name, splits_freq, splits_normalize)

def save_model_results(df: pd.DataFrame, results_name, org_name: str, splits_freq: str, splits_normalize: bool, k_recommendations: list[int]):
    missing_cols = set([*MODEL_RESULTS_COLS, *all_metric_ks(k_recommendations)]).difference(df.columns)
    if missing_cols:
        raise ValueError(f'The following columns should be included {missing_cols}')

    # TODO: Check that the index contains the hparams and fold
    # TODO: Check that the folds are not repeated
    _save_pq(df, f'models/{results_name}', org_name, splits_freq, splits_normalize)

def get_model_results(results_name, org_name: str, splits_freq: str, splits_normalize: bool):
    return _load_pq(f'models/{results_name}', org_name, splits_freq, splits_normalize)

def load_proposals(org_name, base=DEFAULT_DATA_PATH, text=False):
    base = Path(base).expanduser()
    df = pd.read_parquet(base / org_name / 'proposals.pq')
    if not text:
        df.drop(columns=['title', 'description'], inplace=True)
    return df

def load_votes(org_name, base=DEFAULT_DATA_PATH):
    base = Path(base).expanduser()
    df = pd.read_parquet(base / org_name / 'votes.pq')
    return df
