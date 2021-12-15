"""
This file has the purpose of retrieving from Amazon S3 
the requested dataset.

The exposed function is load_data, 
which receives an path in input. 
This path corresponds to a dataset located into the
temporal-summarization bucket on Amazon S3.
"""

import json
import boto3

def download_data(dataset_path: str):
    """
    Receives a string in input (the dataset path)
    and downloads the requested data from Amazon S3.

    If no errors are raised, the downloading phase
    ended correctly.
    """

    with open('./config/rootkey.json') as f:
        config = json.load(f)
        aws_access_key_id     = config['AWSAccessKeyId']
        aws_secret_access_key = config['AWSSecretKey']
        bucket                = config['Bucket']

    print("> Downloading dataset from Amazon S3...")
    s3 = boto3.client('s3', aws_access_key_id = aws_access_key_id , aws_secret_access_key = aws_secret_access_key)
    s3.download_file(bucket, dataset_path, dataset_path)
    print("> Dataset downloaded!")