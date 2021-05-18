import os
import json

import cloudpickle
import numpy as np
import onnxruntime as rt

import ads
from sklearn.preprocessing import LabelEncoder
from dask import dataframe as ddf
import pandas as pd
import traceback
import oci


model_name = 'model.onnx'
transformer_name = 'onnx_data_transformer.json'


"""
   Inference script. This script is used for prediction by scoring server when schema is known.
"""


def load_model(model_file_name=model_name):
    """
    Loads model from the serialized format

    Returns
    -------
    model:  an onnxruntime session instance
    """
    model_dir = os.path.dirname(os.path.realpath(__file__))
    contents = os.listdir(model_dir)
    if model_file_name in contents:
        return rt.InferenceSession(os.path.join(model_dir, model_file_name))
    else:
        raise Exception('{0} is not found in model directory {1}'.format(model_file_name, model_dir))


def predict(data, model=load_model()):
    """
    Returns prediction given the model and data to predict

    Parameters
    ----------
    model: Model session instance returned by load_model API
    data: Data format as expected by the onnxruntime API

    Returns
    -------
    predictions: Output from scoring server
        Format: {'prediction':output from model.predict method}

    """
    print('data path', data)
    try:
        df = ddf.read_parquet(
            data,
            engine='pyarrow',
            columns=['rev_text'],
        ).compute()
    except:
        print(traceback.format_exc())
        df = ddf.read_parquet(
            data,
            engine='pyarrow',
            columns=['rev_text'],
            storage_options={
                "config": oci.config.from_file(os.path.join("~/.oci", "config"))
            },
        ).compute()
    df = df[:1000]
    X = df.values
    print("shapes")
    print(X.shape)
    print(df.shape)
    input_data = {'input': X}
    pred = model.run(None, input_data)[0]
    print(len(pred))
    df['pred'] = model.run(None, input_data)[0].tolist()
    print(df.head())
    dask_df = ddf.from_pandas(df, npartitions=1)
    try:
        dask_df.to_csv('oci://jize-dev/jobs-demo/deploy/pred.csv', single_file=True)
    except:
        dask_df.to_csv(
            'oci://jize-dev@ociodscdev/jobs-demo/deploy/pred.csv',
            single_file=True,
            storage_options={
                "config": oci.config.from_file(os.path.join("~/.oci", "config"))
            }
        )
    print("finished writing to object storage")

    return {'output_path': 'oci://jize-dev@ociodscdev/jobs-demo/deploy/pred.csv'}
