import itertools as it

from recommenders.evaluation.python_evaluation import precision_at_k, ndcg_at_k, map_at_k, recall_at_k, r_precision_at_k

metrics_f = {
    'precision': precision_at_k, 
    'ndcg': ndcg_at_k, 
    'map': map_at_k, 
    'recall': recall_at_k,
    'r-precision': r_precision_at_k,
}

def calculate_all_metrics(rating_true, rating_pred, Ks: list[int]) -> dict[str, float]:
    if not isinstance(Ks, list):
        Ks = [Ks]
    
    eval_dict = dict()
    for (name, func), k in it.product(metrics_f.items(), Ks):
        eval_dict[f'{name}@{k}'] = func(rating_true, rating_pred, k=k)

    return eval_dict
