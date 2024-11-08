import paramiko
from pathlib import Path
import logging
from datetime import datetime, timedelta
import time
import xarray as xr
import pandas as pd
from typing import Optional, List
import os
import shutil
import yaml

class DataManager:
    """Manages data downloading and caching for the dashboard."""

    def __init__(self, config_path: str = 'config.yaml'):
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Load configuration
        self.config = self.load_config(config_path)

        # Initialize cache directory
        self.cache_dir = Path(self.config['cache_directory'])
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # SSH connection details
        self.ssh_config = self.config['ssh']

        # Cache settings
        self.cache_max_age = timedelta(hours=self.config['cache_max_age_hours'])
        self.max_cache_size = self.config['max_cache_size_mb'] * 1024 * 1024  # Convert to bytes

    @staticmethod
    def load_config(config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}")

    def get_data(self, date: datetime) -> Optional[xr.Dataset]:
        """Get data for specific date, either from cache or AWS."""
        try:
            filename = f'HS_{date.strftime("%Y%m%d")}.nc'
            cache_file = self.cache_dir / filename

            # Check if we need to update cache
            if self.should_update_cache(cache_file):
                self.logger.info(f"Downloading fresh data for {date}")
                self.download_from_aws(filename, cache_file)
            else:
                self.logger.info(f"Using cached data for {date}")

            # Load and return the data
            return xr.open_dataset(cache_file)

        except Exception as e:
            self.logger.error(f"Failed to get data for {date}: {e}")
            return None

    def should_update_cache(self, cache_file: Path) -> bool:
        """Determine if cache should be updated."""
        if not cache_file.exists():
            return True

        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        return file_age > self.cache_max_age

    def download_from_aws(self, filename: str, cache_file: Path):
        """Download file from AWS with error handling and retries."""
        max_retries = 3
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                # Set up SSH client
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

                # Connect to remote server
                ssh.connect(
                    hostname=self.ssh_config['hostname'],
                    username=self.ssh_config['username'],
                    key_filename=self.ssh_config['key_path']
                )

                # Download file
                sftp = ssh.open_sftp()
                remote_path = f"{self.ssh_config['remote_path']}/{filename}"

                # Download to temporary file first
                temp_file = cache_file.with_suffix('.tmp')
                sftp.get(remote_path, str(temp_file))

                # Move to final location
                temp_file.rename(cache_file)

                self.logger.info(f"Successfully downloaded {filename}")
                break

            except Exception as e:
                self.logger.error(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise

            finally:
                try:
                    sftp.close()
                    ssh.close()
                except:
                    pass

    def clean_cache(self):
        """Clean old files from cache directory."""
        try:
            # Get list of files sorted by modification time
            cache_files = sorted(
                self.cache_dir.glob('*.nc'),
                key=lambda x: x.stat().st_mtime
            )

            # Calculate total size
            total_size = sum(f.stat().st_size for f in cache_files)

            # Remove old files if total size exceeds limit
            while total_size > self.max_cache_size and cache_files:
                oldest_file = cache_files.pop(0)
                total_size -= oldest_file.stat().st_size
                oldest_file.unlink()
                self.logger.info(f"Removed old cache file: {oldest_file}")

            # Remove files older than max age
            cutoff_time = datetime.now() - self.cache_max_age
            for file in cache_files:
                if datetime.fromtimestamp(file.stat().st_mtime) < cutoff_time:
                    file.unlink()
                    self.logger.info(f"Removed expired cache file: {file}")

        except Exception as e:
            self.logger.error(f"Error cleaning cache: {e}")

    def get_available_dates(self) -> List[datetime]:
        """Get list of dates with available data."""
        try:
            files = self.cache_dir.glob('HS_*.nc')
            dates = []
            for file in files:
                try:
                    date_str = file.stem.split('_')[1]
                    dates.append(datetime.strptime(date_str, '%Y%m%d'))
                except:
                    continue
            return sorted(dates)
        except Exception as e:
            self.logger.error(f"Error getting available dates: {e}")
            return []
