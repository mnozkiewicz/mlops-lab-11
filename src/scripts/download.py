import os
import boto3
from .settings import Settings


def download_s3_folder(settings: Settings):
    s3 = boto3.client("s3")

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=settings.s3_bucket_name, Prefix=settings.s3_model_path):
        for obj in page.get("Contents", []):
            s3_key = obj["Key"]

            # Skip "folders"
            if s3_key.endswith("/"):
                continue

            local_path = os.path.join(settings.artifacts_path, s3_key)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(settings.s3_bucket_name, s3_key, local_path)
