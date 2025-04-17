import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Set
import xarray as xr
from dataclasses import dataclass

from utils.data_manager import DataManager

@dataclass
class DataCheck:
    missing_files: List[str]
    incomplete_files: List[str]
    outdated_files: List[str]

class DataChecker:
    """Checks for missing, incomplete, or outdated data files and manages downloads."""

    def __init__(self, data_manager, input_dir: str, days_to_keep: int = 3):
        self.logger = logging.getLogger('snowmapper.checker')
        self.data_manager = data_manager
        self.input_dir = Path(input_dir)
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.days_to_keep = days_to_keep

    def get_expected_files(self) -> Set[str]:
        """Generate list of expected files for the last n days."""
        expected_files = set()
        today = datetime.now()

        for i in range(self.days_to_keep):
            date = today - timedelta(days=i)
            for var_info in self.data_manager.VARIABLES.values():
                filename = f"{var_info.file_prefix}_{date.strftime('%Y%m%d')}.nc"
                expected_files.add(filename)

        return expected_files

    def check_file_validity(self, file_path: Path) -> bool:
        """Check if a file is valid and complete."""
        try:
            with xr.open_dataset(file_path) as ds:
                # Check if file has expected dimensions
                if 'time' not in ds.dims or 'lat' not in ds.dims or 'lon' not in ds.dims:
                    return False

                # Check if file has 10 time steps
                if ds.dims['time'] != 10:
                    return False

                # Basic data check (no complete NaN slices)
                for var in ds.data_vars:
                    if ds[var].isnull().all():
                        return False

                return True
        except Exception as e:
            self.logger.error(f"Error validating file {file_path}: {e}")
            return False

    async def check_data(self) -> DataCheck:
        """Check for missing, incomplete, or outdated files."""
        expected_files = self.get_expected_files()
        existing_files = set(f.name for f in self.input_dir.glob("*.nc"))

        missing_files = []
        incomplete_files = []
        outdated_files = []

        # Check for missing files
        missing_files = list(expected_files - existing_files)

        # Check existing files for completeness and age
        for filename in existing_files:
            file_path = self.input_dir / filename

            # Check if file is outdated
            file_date_str = filename.split('_')[1].split('.')[0]
            file_date = datetime.strptime(file_date_str, '%Y%m%d').date()
            if (datetime.now().date() - file_date).days > self.days_to_keep:
                outdated_files.append(filename)
                continue

            # Check if file is incomplete
            if not self.check_file_validity(file_path):
                incomplete_files.append(filename)

        return DataCheck(missing_files, incomplete_files, outdated_files)

    async def update_data(self):
        """Download missing files and replace incomplete ones."""
        data_check = await self.check_data()
        files_to_download = data_check.missing_files + data_check.incomplete_files

        if files_to_download:
            self.logger.info(f"Downloading {len(files_to_download)} files")
            for filename in files_to_download:
                try:
                    await self.data_manager.download_file(
                        filename,
                        self.input_dir / filename
                    )
                    self.logger.info(f"Successfully downloaded {filename}")
                except Exception as e:
                    self.logger.error(f"Failed to download {filename}: {e}")

        # Clean up outdated files
        for filename in data_check.outdated_files:
            try:
                (self.input_dir / filename).unlink()
                self.logger.info(f"Removed outdated file {filename}")
            except Exception as e:
                self.logger.error(f"Failed to remove {filename}: {e}")

        return {
            'downloaded': len(files_to_download),
            'removed': len(data_check.outdated_files),
            'failed': [],  # Add tracking of failed downloads if needed
        }

async def run_data_check(config):
    """Main function to run data checking and updating."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('snowmapper.main')

    try:
        data_manager = DataManager(config)
        checker = DataChecker(
            data_manager=data_manager,
            input_dir=config['input_dir'],
            days_to_keep=config['retention_days']
        )

        logger.info("Starting data check...")
        result = await checker.update_data()

        logger.info(f"Data check completed: "
                   f"Downloaded {result['downloaded']} files, "
                   f"Removed {result['removed']} outdated files")

        return result

    except Exception as e:
        logger.error(f"Error in data check: {e}")
        raise

