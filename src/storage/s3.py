"""S3-compatible storage backend for production object storage."""

from __future__ import annotations

import hashlib

from botocore.exceptions import ClientError

from src.storage.base import StoredObject


class S3CompatibleStorage:
    def __init__(
        self,
        *,
        bucket: str,
        endpoint_url: str | None,
        region: str,
        access_key_id: str | None,
        secret_access_key: str | None,
    ) -> None:
        import boto3
        from botocore.config import Config

        self._bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            config=Config(s3={"addressing_style": "path"}),
        )

    def put_bytes(self, key: str, content: bytes, *, mime_type: str, sha256: str | None = None) -> StoredObject:
        digest = hashlib.sha256(content).hexdigest()
        if sha256 and digest != sha256:
            raise ValueError("ARTIFACT_HASH_MISMATCH")
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=content,
            ContentType=mime_type,
            Metadata={"sha256": digest},
        )
        return StoredObject(key=key, size_bytes=len(content), mime_type=mime_type, sha256=digest)

    def get_bytes(self, key: str) -> bytes:
        return self._client.get_object(Bucket=self._bucket, Key=key)["Body"].read()

    def delete(self, key: str) -> None:
        self._client.delete_object(Bucket=self._bucket, Key=key)

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise
        return True

    def head(self, key: str) -> StoredObject:
        response = self._client.head_object(Bucket=self._bucket, Key=key)
        return StoredObject(
            key=key,
            size_bytes=int(response["ContentLength"]),
            mime_type=response.get("ContentType"),
            sha256=response.get("Metadata", {}).get("sha256"),
        )

    def list_keys(self, prefix: str) -> list[str]:
        keys: list[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            keys.extend(item["Key"] for item in page.get("Contents", []))
        return keys

    def signed_download_url(self, key: str, expires_seconds: int) -> str:
        return self._client.generate_presigned_url(
            "get_object", Params={"Bucket": self._bucket, "Key": key}, ExpiresIn=expires_seconds
        )

    def signed_upload_url(self, key: str, *, mime_type: str, expires_seconds: int, sha256: str | None) -> str:
        metadata = {"sha256": sha256} if sha256 else {}
        return self._client.generate_presigned_url(
            "put_object",
            Params={"Bucket": self._bucket, "Key": key, "ContentType": mime_type, "Metadata": metadata},
            ExpiresIn=expires_seconds,
            HttpMethod="PUT",
        )
