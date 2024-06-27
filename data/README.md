These files were generated using the [gen_csv notebook](../notebooks/gen_csv.ipynb), from the [DAO Census TFM](https://www.kaggle.com/datasets/daviddavo/daos-census-tfm) dataset.

This is an old version of the [Census of the Ecosystem of Decentralized Autonomous Organizations](https://zenodo.org/records/10794916) dataset, agumented with textual information.

The textual information is the original multiline information obtained from each DAO API, and for this reason we save the data in parquet format. You can read it using `pd.read_parquet`.