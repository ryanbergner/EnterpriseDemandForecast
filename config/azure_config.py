"""Utility helpers for configuring Azure Blob Storage access for Databricks runs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class AzureStorageConfig:
    storage_account: str
    container: str
    blob_path: str
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    tenant_id: Optional[str] = None
    databricks_workspace_url: Optional[str] = None
    databricks_token: Optional[str] = None

    @property
    def abfss_uri(self) -> str:
        clean_blob_path = self.blob_path.lstrip("/")
        return f"abfss://{self.container}@{self.storage_account}.dfs.core.windows.net/{clean_blob_path}"

    @classmethod
    def from_json(cls, path: Path) -> "AzureStorageConfig":
        data = json.loads(Path(path).read_text())
        return cls(**data)


def load_azure_config(config_path: Optional[str] = None) -> AzureStorageConfig:
    if config_path:
        return AzureStorageConfig.from_json(Path(config_path))

    storage_account = os.getenv("M5_AZURE_STORAGE_ACCOUNT")
    container = os.getenv("M5_AZURE_CONTAINER")
    blob_path = os.getenv("M5_AZURE_BLOB_PATH")
    client_id = os.getenv("M5_AZURE_CLIENT_ID")
    client_secret = os.getenv("M5_AZURE_CLIENT_SECRET")
    tenant_id = os.getenv("M5_AZURE_TENANT_ID")
    workspace_url = os.getenv("M5_DATABRICKS_WORKSPACE_URL")

    missing = [name for name, value in [
        ("M5_AZURE_STORAGE_ACCOUNT", storage_account),
        ("M5_AZURE_CONTAINER", container),
        ("M5_AZURE_BLOB_PATH", blob_path),
    ] if not value]
    if missing:
        raise ValueError(
            "Missing required Azure storage configuration. "
            f"Set the following environment variables or provide a config file: {', '.join(missing)}"
        )

    return AzureStorageConfig(
        storage_account=storage_account,
        container=container,
        blob_path=blob_path,
        client_id=client_id,
        client_secret=client_secret,
        tenant_id=tenant_id,
        databricks_workspace_url=workspace_url,
        databricks_token=os.getenv("M5_DATABRICKS_TOKEN"),
    )
