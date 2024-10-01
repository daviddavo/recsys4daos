import os, sys
from pathlib import Path
import json

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from recsys4daos.evaluation import all_metric_ks
import dataclasses
from recsys4daos.utils.notebooks import DaoToRun

DEFAULT_INPUT_PATH = '../data/input'
DEFAULT_OUTPUT_PATH = '../data/output'
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
    fname = _gen_output_fname(prefix, *args, ext='parquet', **kwargs)
    fname.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(fname)
    print(f"Saved dataframe into {fname}")

def _load_pq(prefix, *args, **kwargs):
    fname = _gen_output_fname(prefix, *args, ext='parquet', **kwargs)
    return pd.read_parquet(fname)

def save_dao_datum(dao, key, val):
    base_path = Path(os.getenv('RS4DAO_OUTPUT_PATH', DEFAULT_OUTPUT_PATH)).expanduser()
    path = base_path / 'daos-info.json'

    if path.exists():
        with open(path, 'r') as f:
            data = json.load(f)
    else:
        data = dict()

    if dao not in data:
        data[dao] = dict()

    data[dao][key] = val
    with open(path, 'w') as f:
        json.dumps(data) # Fail if wrong
        json.dump(data, f, indent=4)

def load_daos_data():
    base_path = Path(os.getenv('RS4DAO_OUTPUT_PATH', DEFAULT_OUTPUT_PATH)).expanduser()
    path = base_path / 'daos-info.json'
    with open(path, 'r') as f:
        return json.load(f)

def save_daos_to_run(list_dtr: list[DaoToRun]):
    base_path = Path(os.getenv('RS4DAO_OUTPUT_PATH', DEFAULT_OUTPUT_PATH)).expanduser()
    with (base_path / 'daos-to-run.json').open('w') as f:
        json.dump([dataclasses.asdict(d) for d in list_dtr], f, indent=4)

def load_daos_to_run() -> list[DaoToRun]:
    base_path = Path(os.getenv('RS4DAO_OUTPUT_PATH', DEFAULT_OUTPUT_PATH)).expanduser()
    with (base_path / 'daos-to-run.json').open('r') as f:
        return [ DaoToRun(**d) for d in json.load(f) ]

def save_openpop(df: pd.DataFrame, org_name: str, splits_freq: str, splits_normalize: bool):
    _save_pq(df, 'baseline/openpop', org_name, splits_freq, splits_normalize)

def load_openpop(org_name: str, splits_freq: str, splits_normalize: bool):
    return _load_pq('baseline/openpop', org_name, splits_freq, splits_normalize)

def save_perfect(df: pd.DataFrame, org_name: str, splits_freq: str, splits_normalize: bool):
    _save_pq(df, 'baseline/perfect', org_name, splits_freq, splits_normalize)

def load_perfect(org_name: str, splits_freq: str, splits_normalize: bool):
    return _load_pq('baseline/perfect', org_name, splits_freq, splits_normalize)

def save_folds_info(df: pd.DataFrame, org_name: str, splits_freq: str, splits_normalize: bool):
    _save_pq(df, 'baseline/folds-info', org_name, splits_freq, splits_normalize)

def load_folds_info(org_name: str, splits_freq: str, splits_normalize: bool):
    return _load_pq('baseline/folds-info', org_name, splits_freq, splits_normalize)

def save_model_results(df: pd.DataFrame, results_name, org_name: str, splits_freq: str, splits_normalize: bool, k_recommendations: list[int]):
    missing_cols = set([*MODEL_RESULTS_COLS, *all_metric_ks(k_recommendations)]).difference(df.columns)
    if missing_cols:
        raise ValueError(f'The following columns should be included {missing_cols}')

    assert df.index.names[0] == 'fold', 'The first index should be "fold"'
    # TODO: Check that the index contains the hparams and fold
    # TODO: Check that the folds are not repeated
    # Check that fold is of type datetime
    assert is_datetime64_any_dtype(df.index.dtypes['fold']), 'The fold index should be datetime'
    _save_pq(df, f'models/{results_name}', org_name, splits_freq, splits_normalize)

def get_model_results(results_name, org_name: str, splits_freq: str, splits_normalize: bool):
    return _load_pq(f'models/{results_name}', org_name, splits_freq, splits_normalize)

def load_proposals(org_name, base=DEFAULT_INPUT_PATH, text=False):
    base = Path(base).expanduser()
    df = pd.read_parquet(base / org_name / 'proposals.parquet')
    if not text:
        df.drop(columns=['title', 'description'], inplace=True)
    return df

def load_votes(org_name, base=DEFAULT_INPUT_PATH):
    base = Path(base).expanduser()
    df = pd.read_parquet(base / org_name / 'votes.parquet')
    return df
