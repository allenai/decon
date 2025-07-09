#!/usr/bin/env python3
"""Download evaluation datasets from S3 bucket."""

import os
import boto3
from pathlib import Path


def download_s3_bucket(bucket_name, local_dir):
    """Download all files from an S3 bucket to a local directory."""
    # Create S3 client using environment variables for credentials
    # AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY should be set
    s3 = boto3.client('s3')
    
    # Create local directory if it doesn't exist
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # List all objects in the bucket
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name)
        
        total_files = 0
        for page in pages:
            if 'Contents' not in page:
                continue
                
            for obj in page['Contents']:
                key = obj['Key']
                
                # Skip if it's a directory marker
                if key.endswith('/'):
                    continue
                
                # Create local file path
                local_file_path = os.path.join(local_dir, key)
                
                # Create parent directory if needed
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                
                # Download the file
                print(f"Downloading: {key}")
                s3.download_file(bucket_name, key, local_file_path)
                total_files += 1
        
        print(f"\nSuccessfully downloaded {total_files} files from {bucket_name}")
        
    except Exception as e:
        print(f"Error downloading from S3: {e}")
        raise


if __name__ == "__main__":
    bucket_name = "decon-evals"
    local_dir = "fixtures/reference"
    
    # Check if AWS credentials are set
    if not os.environ.get('AWS_ACCESS_KEY_ID') or not os.environ.get('AWS_SECRET_ACCESS_KEY'):
        print("Error: AWS credentials not found in environment variables.")
        print("Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables.")
        exit(1)
    
    print(f"Downloading files from S3 bucket '{bucket_name}' to '{local_dir}'...")
    download_s3_bucket(bucket_name, local_dir)