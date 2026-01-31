import os
from typing import Optional

try:
    import ibm_boto3
    from ibm_botocore.config import Config
except Exception:
    ibm_boto3 = None
    Config = None


def cos_enabled() -> bool:
    return all([
        os.getenv("COS_ENDPOINT"),
        os.getenv("COS_ACCESS_KEY_ID"),
        os.getenv("COS_SECRET_ACCESS_KEY"),
        os.getenv("COS_BUCKET"),
    ]) and ibm_boto3 is not None and Config is not None


def _get_client():
    if not ibm_boto3 or not Config:
        raise RuntimeError("ibm-cos-sdk not available; install ibm-cos-sdk or ibm-watsonx-ai")
    # Sanitize values to avoid quoted env entries
    endpoint = (os.getenv("COS_ENDPOINT", "https://s3.us-south.cloud-object-storage.appdomain.cloud") or "").strip().strip('"').strip("'")
    access_key = (os.getenv("COS_ACCESS_KEY_ID") or "").strip().strip('"').strip("'")
    secret_key = (os.getenv("COS_SECRET_ACCESS_KEY") or "").strip().strip('"').strip("'")
    if not (access_key and secret_key):
        raise RuntimeError("Missing COS_ACCESS_KEY_ID or COS_SECRET_ACCESS_KEY")
    return ibm_boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint,
        config=Config(signature_version="s3v4"),
    )


def upload_file_to_cos(local_path: str, key: str, bucket: Optional[str] = None) -> str:
    bucket = bucket or os.getenv("COS_BUCKET")
    if not bucket:
        raise RuntimeError("COS_BUCKET not set")
    client = _get_client()
    client.upload_file(local_path, bucket, key)
    return f"s3://{bucket}/{key}"
