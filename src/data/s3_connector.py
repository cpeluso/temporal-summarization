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

def download_file(filename: str):
    """
    Receives a string in input (the file path)
    and downloads the requested file from Amazon S3.

    If no errors are raised, the downloading phase
    ended correctly.
    """

    with open('./config/rootkey.json') as f:
        config = json.load(f)
        aws_access_key_id     = config['AWSAccessKeyId']
        aws_secret_access_key = config['AWSSecretKey']
        bucket                = config['Bucket']

    print(f"> Downloading file {filename} from Amazon S3...")
    s3 = boto3.client('s3', aws_access_key_id = aws_access_key_id , aws_secret_access_key = aws_secret_access_key)
    s3.download_file(bucket, filename, filename)
    print("> File downloaded!")

def upload_file(filename):
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

    print("> Uploading file to Amazon S3...")
    s3 = boto3.client('s3', aws_access_key_id = aws_access_key_id , aws_secret_access_key = aws_secret_access_key)
    response = s3.upload_file(filename, bucket, filename)
    print("> File uploaded!")