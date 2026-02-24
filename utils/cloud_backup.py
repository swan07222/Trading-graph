"""Cloud Backup Module.

Provides optional cloud backup support for:
- Model artifacts
- Database backups
- Configuration files
- Audit logs

Supported providers:
- AWS S3
- Azure Blob Storage
- Google Cloud Storage
- S3-compatible services (Minio, etc.)
"""
from __future__ import annotations

import hashlib
import json
import os
import tarfile
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class BackupConfig:
    """Cloud backup configuration."""
    # Provider settings
    provider: str = "s3"  # s3, azure, gcs
    bucket_name: str = ""
    region: str = "us-east-1"
    
    # Authentication
    access_key: str = ""
    secret_key: str = ""
    endpoint_url: str | None = None  # For S3-compatible services
    
    # Azure-specific
    connection_string: str = ""
    
    # GCP-specific
    credentials_path: str = ""
    project_id: str = ""
    
    # Backup settings
    backup_prefix: str = "trading-graph"
    retention_days: int = 30
    compression: bool = True
    encryption: bool = True
    
    # Schedule
    auto_backup: bool = False
    backup_frequency: str = "daily"  # hourly, daily, weekly
    
    # Last backup
    last_backup_time: datetime | None = None
    last_backup_status: str = ""


@dataclass
class BackupManifest:
    """Backup manifest tracking."""
    backup_id: str = ""
    created_at: datetime | None = None
    
    # Contents
    files: list[dict[str, Any]] = field(default_factory=list)
    total_size: int = 0
    checksum: str = ""
    
    # Metadata
    hostname: str = ""
    username: str = ""
    version: str = "0.1.0"
    
    # Retention
    expires_at: datetime | None = None
    
    def __post_init__(self) -> None:
        if not self.backup_id:
            self.backup_id = f"BACKUP_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'backup_id': self.backup_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'files': self.files,
            'total_size': self.total_size,
            'checksum': self.checksum,
            'hostname': self.hostname,
            'username': self.username,
            'version': self.version,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
        }


class CloudBackup:
    """Cloud backup manager for trading system data.
    
    Usage:
        backup = CloudBackup(config)
        
        # Manual backup
        backup.create_backup()
        
        # Restore from backup
        backup.restore(backup_id)
        
        # List available backups
        backups = backup.list_backups()
    """
    
    def __init__(self, config: BackupConfig) -> None:
        """Initialize cloud backup.
        
        Args:
            config: Backup configuration
        """
        self.config = config
        self._client: Any = None
        self._initialized = False
    
    def _init_client(self) -> None:
        """Initialize cloud storage client."""
        if self._initialized:
            return
        
        provider = self.config.provider.lower()
        
        try:
            if provider == "s3":
                self._init_s3_client()
            elif provider == "azure":
                self._init_azure_client()
            elif provider == "gcs":
                self._init_gcs_client()
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            self._initialized = True
            log.info(f"Cloud backup client initialized ({provider})")
        except ImportError as e:
            log.warning(f"Cloud backup dependencies not installed: {e}")
        except Exception as e:
            log.error(f"Failed to initialize cloud backup: {e}")
    
    def _init_s3_client(self) -> None:
        """Initialize S3 client."""
        try:
            import boto3
            from botocore.config import Config
            
            # Build configuration
            client_config = Config(
                region_name=self.config.region,
                retries={'max_attempts': 3}
            )
            
            # Create client
            self._client = boto3.client(
                's3',
                aws_access_key_id=self.config.access_key,
                aws_secret_access_key=self.config.secret_key,
                endpoint_url=self.config.endpoint_url,
                config=client_config,
            )

            # Ensure bucket exists
            self._ensure_bucket()

        except ImportError as e:
            raise ImportError("Install boto3: pip install boto3") from e

    def _init_azure_client(self) -> None:
        """Initialize Azure Blob Storage client."""
        try:
            from azure.storage.blob import BlobServiceClient

            if self.config.connection_string:
                self._client = BlobServiceClient.from_connection_string(
                    self.config.connection_string
                )
            else:
                raise ValueError("Azure connection_string required")

        except ImportError as e:
            raise ImportError("Install azure-storage-blob: pip install azure-storage-blob") from e

    def _init_gcs_client(self) -> None:
        """Initialize Google Cloud Storage client."""
        try:
            from google.cloud import storage

            if self.config.credentials_path:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.config.credentials_path

            self._client = storage.Client(project=self.config.project_id)

        except ImportError as e:
            raise ImportError("Install google-cloud-storage: pip install google-cloud-storage") from e
    
    def _ensure_bucket(self) -> None:
        """Ensure bucket exists (S3)."""
        if self.config.provider.lower() != "s3":
            return
        
        try:
            self._client.head_bucket(Bucket=self.config.bucket_name)
        except Exception:
            # Create bucket
            self._client.create_bucket(
                Bucket=self.config.bucket_name,
                CreateBucketConfiguration={'LocationConstraint': self.config.region}
            )
            log.info(f"Created bucket: {self.config.bucket_name}")
    
    def create_backup(
        self,
        directories: list[Path | str] | None = None,
        manifest: BackupManifest | None = None,
    ) -> BackupManifest | None:
        """Create cloud backup.
        
        Args:
            directories: Directories to backup (default: data, models, config)
            manifest: Optional manifest to use
        
        Returns:
            BackupManifest if successful, None otherwise
        """
        self._init_client()
        
        if not self._initialized:
            log.error("Cloud backup client not initialized")
            return None
        
        # Default directories
        if directories is None:
            from config.settings import CONFIG
            directories = [
                CONFIG.data_dir,
                CONFIG.model_dir,
                CONFIG.log_dir,
            ]
        
        # Create temporary archive
        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
            archive_path = Path(tmp.name)
        
        try:
            # Create tarball
            self._create_archive(archive_path, directories)
            
            # Upload to cloud
            manifest = manifest or BackupManifest()
            manifest.total_size = archive_path.stat().st_size
            manifest.checksum = self._calculate_checksum(archive_path)
            
            object_key = self._get_object_key(manifest.backup_id)
            self._upload_file(archive_path, object_key, manifest)
            
            # Save manifest
            self._save_manifest(manifest)
            
            # Cleanup
            archive_path.unlink()
            
            # Update config
            self.config.last_backup_time = datetime.now()
            self.config.last_backup_status = "success"
            
            log.info(f"Backup created: {manifest.backup_id}")
            return manifest
            
        except Exception as e:
            log.error(f"Backup failed: {e}")
            self.config.last_backup_status = f"failed: {e}"
            
            # Cleanup on failure
            if archive_path.exists():
                archive_path.unlink()
            
            return None
    
    def _create_archive(
        self,
        archive_path: Path,
        directories: list[Path | str],
    ) -> None:
        """Create compressed archive."""
        with tarfile.open(archive_path, 'w:gz') as tar:
            for dir_path in directories:
                path = Path(dir_path)
                if path.exists():
                    # Add directory to archive
                    tar.add(
                        path,
                        arcname=path.name,
                        filter=self._tar_filter,
                    )
                    log.debug(f"Added {path} to archive")
    
    def _tar_filter(
        self,
        tarinfo: tarfile.TarInfo,
    ) -> tarfile.TarInfo | None:
        """Filter archive entries."""
        # Skip cache directories
        skip_dirs = {'__pycache__', '.pytest_cache', '.git', 'venv'}
        for skip in skip_dirs:
            if skip in tarinfo.path:
                return None
        
        # Skip large log files
        if tarinfo.name.endswith('.log') and tarinfo.size > 100 * 1024 * 1024:
            return None
        
        return tarinfo
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _get_object_key(self, backup_id: str) -> str:
        """Get cloud object key."""
        date_str = datetime.now().strftime('%Y/%m/%d')
        return f"{self.config.backup_prefix}/{date_str}/{backup_id}.tar.gz"
    
    def _upload_file(
        self,
        file_path: Path,
        object_key: str,
        manifest: BackupManifest,
    ) -> None:
        """Upload file to cloud storage."""
        provider = self.config.provider.lower()
        
        if provider == "s3":
            self._client.upload_file(
                str(file_path),
                self.config.bucket_name,
                object_key,
                ExtraArgs={
                    'Metadata': {
                        'backup-id': manifest.backup_id,
                        'checksum': manifest.checksum,
                    },
                    'ServerSideEncryption': 'AES256' if self.config.encryption else '',
                }
            )
        elif provider == "azure":
            blob_client = self._client.get_blob_client(
                container=self.config.bucket_name,
                blob=object_key,
            )
            with open(file_path, 'rb') as data:
                blob_client.upload_blob(data, overwrite=True)
        elif provider == "gcs":
            bucket = self._client.bucket(self.config.bucket_name)
            blob = bucket.blob(object_key)
            blob.upload_from_filename(str(file_path))
    
    def _save_manifest(self, manifest: BackupManifest) -> None:
        """Save backup manifest."""
        provider = self.config.provider.lower()
        manifest_key = f"{self.config.backup_prefix}/manifests/{manifest.backup_id}.json"
        
        manifest_data = json.dumps(manifest.to_dict(), indent=2)
        
        if provider == "s3":
            self._client.put_object(
                Bucket=self.config.bucket_name,
                Key=manifest_key,
                Body=manifest_data.encode('utf-8'),
                ContentType='application/json',
            )
        elif provider == "azure":
            blob_client = self._client.get_blob_client(
                container=self.config.bucket_name,
                blob=manifest_key,
            )
            blob_client.upload_blob(manifest_data, overwrite=True)
        elif provider == "gcs":
            bucket = self._client.bucket(self.config.bucket_name)
            blob = bucket.blob(manifest_key)
            blob.upload_from_string(manifest_data, content_type='application/json')
    
    def list_backups(self, limit: int = 100) -> list[BackupManifest]:
        """List available backups."""
        self._init_client()
        
        if not self._initialized:
            return []
        
        manifests = []
        prefix = f"{self.config.backup_prefix}/manifests/"
        
        try:
            if self.config.provider.lower() == "s3":
                response = self._client.list_objects_v2(
                    Bucket=self.config.bucket_name,
                    Prefix=prefix,
                    MaxKeys=limit,
                )
                
                for obj in response.get('Contents', []):
                    if obj['Key'].endswith('.json'):
                        manifest = self._load_manifest(obj['Key'])
                        if manifest:
                            manifests.append(manifest)
            
            # Sort by creation time
            manifests.sort(key=lambda m: m.created_at or datetime.min, reverse=True)
            
        except Exception as e:
            log.error(f"Failed to list backups: {e}")
        
        return manifests
    
    def _load_manifest(self, manifest_key: str) -> BackupManifest | None:
        """Load manifest from cloud storage."""
        try:
            if self.config.provider.lower() == "s3":
                
                response = self._client.get_object(
                    Bucket=self.config.bucket_name,
                    Key=manifest_key,
                )
                data = json.loads(response['Body'].read().decode('utf-8'))
            else:
                # Other providers - simplified
                return None
            
            return BackupManifest(
                backup_id=data.get('backup_id', ''),
                created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else None,
                files=data.get('files', []),
                total_size=data.get('total_size', 0),
                checksum=data.get('checksum', ''),
            )
        except Exception as e:
            log.warning(f"Failed to load manifest: {e}")
            return None
    
    def restore(
        self,
        backup_id: str,
        target_dir: Path | str | None = None,
    ) -> bool:
        """Restore from backup."""
        self._init_client()
        
        if not self._initialized:
            return False
        
        # Download archive
        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
            archive_path = Path(tmp.name)
        
        try:
            # Download from cloud
            object_key = f"{self.config.backup_prefix}/{backup_id}.tar.gz"
            self._download_file(archive_path, object_key)
            
            # Extract
            extract_dir = Path(target_dir) if target_dir else Path.cwd()
            extract_dir.mkdir(parents=True, exist_ok=True)
            
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(path=extract_dir)
            
            log.info(f"Restored backup {backup_id} to {extract_dir}")
            return True
            
        except Exception as e:
            log.error(f"Restore failed: {e}")
            return False
        finally:
            if archive_path.exists():
                archive_path.unlink()
    
    def _download_file(self, local_path: Path, object_key: str) -> None:
        """Download file from cloud storage."""
        provider = self.config.provider.lower()
        
        if provider == "s3":
            self._client.download_file(
                self.config.bucket_name,
                object_key,
                str(local_path),
            )
        elif provider == "azure":
            blob_client = self._client.get_blob_client(
                container=self.config.bucket_name,
                blob=object_key,
            )
            with open(local_path, 'wb') as download_file:
                download_file.write(blob_client.download_blob().readall())
        elif provider == "gcs":
            bucket = self._client.bucket(self.config.bucket_name)
            blob = bucket.blob(object_key)
            blob.download_to_filename(str(local_path))
    
    def delete_backup(self, backup_id: str) -> bool:
        """Delete backup."""
        self._init_client()
        
        if not self._initialized:
            return False
        
        try:
            object_key = f"{self.config.backup_prefix}/{backup_id}.tar.gz"
            manifest_key = f"{self.config.backup_prefix}/manifests/{backup_id}.json"
            
            if self.config.provider.lower() == "s3":
                self._client.delete_object(
                    Bucket=self.config.bucket_name,
                    Key=object_key,
                )
                self._client.delete_object(
                    Bucket=self.config.bucket_name,
                    Key=manifest_key,
                )
            
            log.info(f"Deleted backup {backup_id}")
            return True
            
        except Exception as e:
            log.error(f"Delete failed: {e}")
            return False
    
    def cleanup_old_backups(self) -> int:
        """Clean up backups older than retention period."""
        deleted = 0
        cutoff = datetime.now() - timedelta(days=self.config.retention_days)
        
        for manifest in self.list_backups(limit=1000):
            if manifest.created_at and manifest.created_at < cutoff:
                if self.delete_backup(manifest.backup_id):
                    deleted += 1
        
        log.info(f"Cleaned up {deleted} old backups")
        return deleted


def create_backup(
    provider: str = "s3",
    bucket_name: str = "",
    access_key: str = "",
    secret_key: str = "",
) -> BackupManifest | None:
    """Convenience function to create backup."""
    config = BackupConfig(
        provider=provider,
        bucket_name=bucket_name,
        access_key=access_key,
        secret_key=secret_key,
    )
    backup = CloudBackup(config)
    return backup.create_backup()
