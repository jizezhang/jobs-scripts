import argparse
import os
import traceback

import ads
import oci
import xgboost
import numpy as np
from dask import dataframe as ddf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if "OUTPUT_DIR" not in os.environ:
    os.environ["OUTPUT_DIR"] = "./output"

if "N_ROUNDS" not in os.environ:
    os.environ["N_ROUNDS"] = "10"


def main(logger, data_path):
    logger.log(f"data from {data_path}")
    try:  # job instance has rp
        review_df_full = ddf.read_parquet(
            data_path,
            engine="pyarrow",
            columns=["rev_text", "filtered_text", "overall"],
        ).compute()
    except:  # for local testing with config
        logger.log(traceback.format_exc())
        review_df_full = ddf.read_parquet(
            data_path,
            engine="pyarrow",
            columns=["rev_text", "filtered_text", "overall"],
            storage_options={
                "config": oci.config.from_file(os.path.join("~/.oci", "config"))
            },
        ).compute()

    review_df = review_df_full.sample(frac=0.2)
    logger.log("finished loading data")

    tf_vectorizer = TfidfVectorizer(lowercase=False, max_features=20000)
    tf_vectorizer.fit(review_df["rev_text"])
    logger.log("finished fitting vectorizer")

    trainx_orig, testx_orig, trainy, testy = train_test_split(
        review_df, review_df["overall"].values, test_size=0.20, random_state=42
    )
    logger.log("finished train test split")

    class Callback(xgboost.callback.TrainingCallback):
        def __init__(self, logger):
            self.logger = logger

        def _get_key(self, data, metric):
            return f"{data}-{metric}"

        def after_iteration(self, model, epoch, evals_log):
            print("called")
            for data, metric in evals_log.items():
                for metric_name, log in metric.items():
                    key = self._get_key(data, metric_name)
                    self.logger.log(f"{key}: {log}")
            self.logger.log(f"epoch {epoch}")
            return False

    trainx = tf_vectorizer.transform(trainx_orig["rev_text"])
    testx = tf_vectorizer.transform(testx_orig["rev_text"])

    logger.log(f'xgboost version {xgboost.__version__}')
    d_train = xgboost.DMatrix(trainx, trainy)
    d_val = xgboost.DMatrix(testx, testy)
    xgboost.train(
        {"objective": "multi:softmax", "num_class": 6},
        d_train,
        num_boost_round=int(os.environ['N_ROUNDS']),
        evals=[(d_train, "Train"), (d_val, "Valid")],
        callbacks=[Callback(logger)],
    )

    # model = xgboost.train({"objective": "multi:softmax", "num_class": 6}, d_train)
    # model.save_model('model')
    # for i in range(int(os.environ['N_ROUNDS'])):
    #     model = xgboost.train({"objective": "multi:softmax", "num_class": 6}, d_train, xgb_model='model')
    #     logger.log(f"==== iteration: {i} accuracy: {max(1, accuracy_score(testy, model.predict(d_val)) + np.random.rand() * .3)} ==== ")
    #     model.save_model('model')

    logger.log("finished training model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", default="oci://jize-dev@ociodscdev/jobs/etl-out/*.parquet"
    )
    parser.add_argument("--model-path", default=os.environ["OUTPUT_DIR"])
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=logging.INFO)

    class Logger:
        def __init__(self):
            self.logger = logging.getLogger(__name__)

        def log(self, msg):
            self.logger.info(msg)

    main(Logger(), args.data_path)
