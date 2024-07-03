import warnings
import itertools as it

import numpy as np
import pandas as pd
from recommenders.evaluation.python_evaluation import precision_at_k, ndcg_at_k, map_at_k, recall_at_k, r_precision_at_k

from .utils import Timer

metrics_f = {
    'precision': precision_at_k, 
    'ndcg': ndcg_at_k, 
    'map': map_at_k, 
    'recall': recall_at_k,
    'r-precision': r_precision_at_k,
}

def all_metric_ks(Ks: list[int]):
    if not isinstance(Ks, list):
        Ks = [Ks]
    return ( f'{name}@{k}' for name,k in it.product(metrics_f.keys(), Ks) )

def calculate_all_metrics(
    rating_true,
    rating_pred,
    Ks: list[int],
    **kwargs,
) -> dict[str, float]:
    if not isinstance(Ks, list):
        Ks = [Ks]
    
    with Timer() as t_eval:
        eval_dict = dict()
        for (name, func), k in it.product(metrics_f.items(), Ks):
            eval_dict[f'{name}@{k}'] = func(rating_true, rating_pred, k=k, **kwargs)

    eval_dict['time_eval'] = t_eval.time
    return eval_dict

def test_with_hparams_lenskit(
    algo, 
    fold, 
    k_recommendations: list[int], 
    window_size=None, 
    col_user='user', 
    col_item='item'
) -> dict[str: float]:
    # Get and filter train data
    train = fold.train
    
    if window_size:
        offset = pd.tseries.frequencies.to_offset(window_size)
        train = train[train['timestamp'] > (fold.end - offset)]

    with Timer() as t_train:
        algo.fit(train)

    # TODO: For each user, make the recommendations
    # and then generate a microsoft-like dataframe
    with Timer() as t_rec:
        users = set(fold.test[col_user].unique()).intersection(train[col_user].unique())
        voted_props = train.groupby(col_user)[col_item].unique()
        def _recu(u):
            # Remove proposals the user voted in
            ps = np.setdiff1d(fold.open_proposals, voted_props.loc[u])
            # TODO: WHY DOES IT RETURN SO MANY NAs?
            x = (algo
                .predict_for_user(u, ps)
                .reset_index()
                .rename(columns={'index':col_item, 0:'prediction'})
                # .dropna()
                .fillna(0.00)
                .assign(user=u)[[col_user, col_item, 'prediction']]
            )
            return x
    
        # TODO: Use lenskit.batch.recommend
        # https://lkpy.lenskit.org/en/stable/batch.html#recommendation
        if users:
            recs = pd.concat(map(_recu, users))
        else:
            warnings.warn(f"No users to recommend to with window_size {window_size}", RuntimeWarning)
            recs = pd.DataFrame(columns=[col_item, col_user, 'prediction'])

    return { 
        'fold_t': fold.end,
        'time_train': t_train.time,
        'time_rec': t_rec.time,
        'open_proposals': len(fold.open_proposals),
        'min_recs': recs.groupby(col_user).size().min(),
        'avg_recs': recs.groupby(col_user).size().mean(),
        **calculate_all_metrics(
            fold.test, recs, k_recommendations, col_user=col_user, col_item=col_item,
        )
    }
