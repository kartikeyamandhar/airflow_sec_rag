"""S3-compatible storage backend, used for Cloudflare R2."""

from __future__ import annotations

from typing import TYPE_CHECKING

import boto3
from botocore.exceptions import ClientError

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client

# R2 ignores the AWS region but the SDK still requires one; "auto" is conventional.
_R2_REGION = "auto"
_NOT_FOUND_CODES = frozenset({"404", "NoSuchKey", "NotFound"})


class S3RawStore:
    """A :class:`~app.storage.base.RawStore` backed by S3 / Cloudflare R2."""

    def __init__(
        self,
        *,
        bucket: str,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
        region: str = _R2_REGION,
    ) -> None:
        self._bucket = bucket
        self._client: S3Client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )

    def put(self, key: str, data: bytes, *, content_type: str) -> str:
        self._client.put_object(Bucket=self._bucket, Key=key, Body=data, ContentType=content_type)
        return self.uri_for(key)

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in _NOT_FOUND_CODES:
                return False
            raise
        return True

    def get(self, key: str) -> bytes:
        response = self._client.get_object(Bucket=self._bucket, Key=key)
        return response["Body"].read()

    def uri_for(self, key: str) -> str:
        return f"s3://{self._bucket}/{key}"
