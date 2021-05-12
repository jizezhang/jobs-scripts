import argparse
import importlib
import os
import pathlib
import shutil
import traceback

import ads
import fsspec
import oci
from dask import dataframe as ddf
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import StringTensorType
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

if "OUTPUT_DIR" not in os.environ:
    os.environ["OUTPUT_DIR"] = "./output"


def main(logger, data_path, model_path=os.environ["OUTPUT_DIR"]):
    logger.log(f"data from {data_path}")
    try:
        review_df_full = ddf.read_parquet(
            data_path,
            engine="pyarrow",
            columns=["rev_text", "filtered_text", "overall"],
        ).compute()
    except:
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

    params = {
        "booster": "dart",
        "lambda": 0.6909908717925198,
        "alpha": 0.0031974003042527653,
        "max_depth": 2,
        "eta": 0.8110635088378954,
        "gamma": 0.012888193236050949,
        "grow_policy": "lossguide",
        "sample_type": "uniform",
        "normalize_type": "tree",
        "rate_drop": 0.0037771727376729345,
        "skip_drop": 5.207598767315119e-06,
        "n_estimators": 1,
    }
    xgb_model = XGBClassifier(**params)
    xgb_pipeline = Pipeline([("vectorizer", tf_vectorizer), ("model", xgb_model)]).fit(
        trainx_orig["rev_text"], trainy
    )

    logger.log("finished training model")

    update_registered_converter(
        XGBClassifier,
        "XGBoostXGBClassifier",
        calculate_linear_classifier_output_shapes,
        convert_xgboost,
        options={"nocl": [True, False], "zipmap": [True, False]},
    )
    try:
        model_onnx = convert_sklearn(
            xgb_pipeline, "xgboost", [("input", StringTensorType())], target_opset=12
        )
    except:
        logger.log(traceback.format_exc())

    logger.log("finished converting to onnx")

    # try:
    #     with fsspec.open(model_path, mode="wb") as f:
    #         f.write(model_onnx.SerializeToString())
    # except:
    #     logger.log(traceback.format_exc())
    #     with fsspec.open(
    #         model_path,
    #         mode="wb",
    #         config=oci.config.from_file(os.path.join("~/.oci", "config")),
    #     ) as f:
    #         f.write(model_onnx.SerializeToString())
    #
    # logger.log(f"model saved to {model_path}")

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    pathlib.Path(model_path).mkdir(exist_ok=True, parents=True)
    shutil.copy2(
        os.path.join(curr_dir, "score.py"), os.path.join(model_path, "score.py")
    )

    with fsspec.open(os.path.join(model_path, "model.onnx"), mode="wb") as f:
        f.write(model_onnx.SerializeToString())

    spec = importlib.util.spec_from_file_location(
        "score", os.path.join(model_path, "score.py")
    )
    score = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score)
    score.load_model()

    logger.log(score.predict(data_path))

    # sess = rt.InferenceSession(os.path.join(model_path, "model.onnx"))
    # pred_onx = sess.run(None, {"input": testx_orig[:5].values})
    # logger.log(f"predict {pred_onx[0]}")
    # logger.log(f"predict_proba {pred_onx[1][:1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path")
    parser.add_argument("--model-path", default=os.environ["OUTPUT_DIR"])
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=logging.INFO)

    class Logger:
        def __init__(self):
            self.logger = logging.getLogger(__name__)

        def log(self, msg):
            self.logger.info(msg)

    main(Logger(), args.data_path, args.model_path)
