import os
import traceback
import uuid

import ads
import oci
import onnxruntime as rt
from dask import dataframe as ddf

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
    dask_df = ddf.from_pandas(df, npartitions=1)
    uid = str(uuid.uuid4())
    # try:
    dask_df.to_csv(f'oci://jize-dev@ociodscdev/jobs-demo/deploy/pred-{uid}.csv', single_file=True)
    # except:
    #     print(traceback.format_exc())
    #     dask_df.to_csv(
    #         f'oci://jize-dev@ociodscdev/jobs-demo/deploy/pred-{uid}.csv',
    #         single_file=True,
    #         storage_options={
    #             "config": oci.config.from_file(os.path.join("~/.oci", "config"))
    #         }
    #     )
    print("finished writing to object storage")

    return {'output_path': f'oci://jize-dev@ociodscdev/jobs-demo/deploy/pred-{uid}.csv'}
