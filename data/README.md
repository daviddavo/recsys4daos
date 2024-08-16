## Input folder

The input files were generated using the [gen_input notebook](../notebooks/gen_input.ipynb), from the [DAO Census TFM](https://www.kaggle.com/datasets/daviddavo/daos-census-tfm) dataset.

This is an old version of the [Census of the Ecosystem of Decentralized Autonomous Organizations](https://zenodo.org/records/10794916) dataset, agumented with textual information.

The textual information is the original multiline information obtained from each DAO API, and for this reason we save the data in parquet format. You can read it using `pd.read_parquet`.

## Output folder

The output files were generated using the [run_one notebook](../notebooks/run_one.ipynb), which was run for all daos using the [run_all notebook](../notebooks/run_all.ipynb).

These notebooks just run the other notebooks that start with a number in the [notebooks folder](../notebooks/)

For each organization, you can find two folders: `baseline` and `models`. The first folder contains _parquet files_ with ALL THE FOLDS of two baselines: a perfect recommender that recommends the ground truth, and OpenPop, which recommends the most popular open proposal at the moment of recommendation (fold).

In the `models` folder you will find three parquet files for each model:
- `<model>-best-avg_*`: The results of using the hyperparameters that gets the best average results among all the folds. The hyperparameters will be the same on every fold.
- `<model>-best-val_*`: The results of using in each fold the hyperparameters that get the best results for that fold.
- `<model>-best-test_*`: The results of using in each fold the hyperparameters that got the best results on the previous fold. The hyperparameters will be the same as in `best-val`, but there is one less fold.
- `<model>-all_*`: Includes the results of testing every hyperparameter on each fold.

These files have set the index to the fold-index and the other hyperparameters. The columns are the results for that fold and hyperparameters (time used to train/eval, metrics at different k's, etc.). While the index is a multi-index, the columns are not multi-indexed. Nevertheless, the metrics truncated at different `k` values can be parsed as they all have the name `<metric>@<k>`.
