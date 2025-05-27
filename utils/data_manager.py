import asyncio
import asyncssh
import xarray as xr
import boto3
from pathlib import Path
import logging
from datetime import datetime, timedelta
import os
from typing import Optional, Dict, List, Any
import yaml
import numpy as np
import time

class DataVariable:
    def __init__(self, name: str, file_prefix: str, units: str):
        self.name = name
        self.file_prefix = file_prefix
        self.units = units


class DataManager:
    """Manages data downloading and caching for both local and AWS environments."""

    VARIABLES = {
        'hs': DataVariable('snow_height', 'HS', 'mm'),
        'swe': DataVariable('swe', 'SWE', 'mm'),
        'rof': DataVariable('runoff', 'ROF', 'm3/s')
    }

    def __init__(self, config: Dict[str, Any]):
        """Initialize DataManager with provided configuration.

        Args:
            config: Dictionary containing configuration settings
        """
        self.logger = logging.getLogger('snowmapper.data')
        self.env = os.getenv('DASHBOARD_ENV', 'local')
        self.config = config

        # Initialize variables from config instead of hardcoding
        self.VARIABLES = self._init_variables_from_config()

        # Pring debugging info
        self.logger.debug(f"DataManager initialized in {self.env} environment")
        self.logger.debug(f"Config: {self.config}")
        self.logger.debug(f"Initialized variables: {list(self.VARIABLES.keys())}")

        # Get cache directory from config
        self.cache_dir = Path(self.config['paths']['cache_dir'])
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize appropriate client based on environment
        if self.env == 'aws':
            self.s3 = boto3.client('s3')

        self.logger.info(f"DataManager initialized in {self.env} environment")

    def _init_variables_from_config(self) -> Dict[str, DataVariable]:
        """Initialize variables from configuration.
        
        Returns:
            Dictionary mapping variable codes to DataVariable objects
        """
        variables = {}
        
        if 'variables' not in self.config:
            self.logger.warning("No 'variables' section found in configuration, using defaults")
            # Fallback to defaults if not in config
            return {
                'hs': DataVariable('snow_height', 'HS', 'mm'),
                'swe': DataVariable('swe', 'SWE', 'mm'),
                'rof': DataVariable('runoff', 'ROF', 'm3/s')
            }
            
        for var_code, var_config in self.config['variables'].items():
            name = var_config.get('name', var_code)
            file_prefix = var_config.get('file_prefix', var_code.upper())
            units = var_config.get('units', '')
            
            variables[var_code] = DataVariable(name, file_prefix, units)
            self.logger.debug(f"Initialized variable {var_code}: {name} ({file_prefix}, {units})")
            
        return variables

    async def  get_data_for_date(self, variable: str, date: datetime) -> Optional[xr.Dataset]:
        """Get data for specific variable and date."""
        self.logger.debug(f"Getting data for {variable} on {date}")
        
        if variable not in self.VARIABLES:
            self.logger.error(f"Unknown variable: {variable}. Available variables: {list(self.VARIABLES.keys())}")
            return None
            
        var_info = self.VARIABLES[variable]
        filename = f"{var_info.file_prefix}_{date.strftime('%Y%m%d')}.nc"
        data = await self._get_cached_or_download(filename)
        
        if data is None:
            self.logger.warning(f"No data available for {variable} on {date}")
        else:
            self.logger.debug(f"Successfully loaded data for {variable} on {date}")
        return data

    def should_update_cache(self, cache_file: Path) -> bool:
        """Determine if cache should be updated."""
        if not cache_file.exists():
            return True

        # If the presenta file is older than max_age days, update it
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        max_age = timedelta(hours=self.config['cache']['max_age_hours'])
        return file_age > max_age

    async def get_change_data(self, variable: str, date: datetime, hours: int) -> Optional[xr.Dataset]:
        """Calculate change in variable over specified hours."""
        future_date = date + timedelta(hours=hours)

        current_data = await self.get_data_for_date(variable, date)
        #future_data = await self.get_data_for_date(variable, future_date)

        if current_data is None or future_data is None:
            self.logger.warning(f"Cannot calculate change for {variable} on {date}")
            self.logger.debug(f"Current data: {current_data}")
            self.logger.debug(f"Future data: {future_data}")
            return None

        try:
            change = future_data[variable] - current_data[variable]
            return change
        except Exception as e:
            self.logger.error(f"Error calculating change: {e}")
            return None

    async def _get_cached_or_download(self, filename: str) -> Optional[xr.Dataset]:
        """Get data from cache or download if needed."""
        cache_file = self.cache_dir / filename
        self.logger.debug(f"Checking cache for {filename}")

        if self.should_update_cache(cache_file):
            self.logger.debug(f"Cache needs update for {filename}")
            try:
                await self.download_file(filename, cache_file)
            except Exception as e:
                self.logger.error(f"Failed to download {filename}: {e}")
                if not cache_file.exists():
                    return None

        try:
            data = xr.open_dataset(cache_file)
            self.logger.debug(f"Successfully opened {filename}")
            # Print dimensions and variables
            self.logger.debug(f"Dimensions: {data.dims}")
            self.logger.debug(f"Variables: {data.data_vars}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to open {filename}: {e}")
            return None


    async def download_file(self, filename: str, cache_file: Path):
        max_retries = 3
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                temp_file = cache_file.with_suffix('.tmp')
                async with asyncssh.connect(
                    host=self.config['ssh']['hostname'],
                    username=self.config['ssh']['username'],
                    client_keys=[self.config['ssh']['key_path']],
                    known_hosts=None
                ) as conn:
                    async with conn.start_sftp_client() as sftp:
                        remote_path = f"{self.config['ssh']['remote_path']}/{filename}"
                        self.logger.info(f"Downloading {filename} from {remote_path}")
                        await sftp.get(remote_path, str(temp_file))
                        temp_file.rename(cache_file)
                        self.logger.info(f"Successfully downloaded {filename}")
                        return

            except asyncssh.Error as e:
                if 'No such file' in str(e) or 'File not found' in str(e):
                    self.logger.error(f"File not found: {filename}")
                    raise

                if attempt == max_retries - 1:
                    raise
                self.logger.error(f"SSH operation failed: {str(e)}")
                await asyncio.sleep(retry_delay)

            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(retry_delay)


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

            # Get max size from config (convert MB to bytes)
            max_cache_size = self.config['cache']['max_size_mb'] * 1024 * 1024

            # Get max age from config
            max_age = timedelta(hours=self.config['cache']['max_age_hours'])

            # Remove old files if total size exceeds limit
            while total_size > max_cache_size and cache_files:
                oldest_file = cache_files.pop(0)
                total_size -= oldest_file.stat().st_size
                oldest_file.unlink()
                self.logger.info(f"Removed old cache file: {oldest_file}")

            # Remove files older than max age
            cutoff_time = datetime.now() - max_age
            for file in cache_files:
                if datetime.fromtimestamp(file.stat().st_mtime) < cutoff_time:
                    file.unlink()
                    self.logger.info(f"Removed expired cache file: {file}")

        except Exception as e:
            self.logger.error(f"Error cleaning cache: {e}")

    def get_available_dates(self, variable: Optional[str] = None) -> List[datetime]:
        """Get list of dates with available data.

        Args:
            variable: Optional variable name. If None, checks HS files.
        """
        try:
            if variable:
                var_info = self.VARIABLES[variable]
                pattern = f"{var_info.file_prefix}_*.nc"
            else:
                pattern = "HS_*.nc"  # Default to HS if no variable specified

            dates = []
            for file in self.cache_dir.glob(pattern):
                try:
                    date_str = file.stem.split('_')[1]
                    dates.append(datetime.strptime(date_str, '%Y%m%d'))
                except:
                    continue
            return sorted(dates)
        except Exception as e:
            self.logger.error(f"Error getting available dates: {e}")
            return []
