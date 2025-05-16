import asyncio
import logging
import os
from pathlib import Path
import paramiko
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class TextFileInfo:
    remote_path: str
    local_filename: str

class TextFileDownloader:
    """Downloads text files from a remote server using SCP."""

    def __init__(self, config: Dict, output_dir: str):
        self.logger = logging.getLogger('snowmapper.text_downloader')
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # SSH connection details
        self.ssh_key_path = config['ssh']['key_path']
        self.ssh_host = config['ssh']['hostname']
        self.ssh_user = config['ssh']['username']
        self.remote_path = config['ssh']['remote_path']
        self.climatology_dir = config['climatology']['climatology_dir']
        self.current_file = config['climatology']['current_file']
        self.climate_file = config['climatology']['climate_file']
        self.previous_file = config['climatology']['previous_file']

        # Throw an error if SSH details are not provided
        if not self.ssh_key_path or not self.ssh_host or not self.ssh_user:
            raise ValueError("SSH connection details are not fully provided in the config.")
        
        # Files to download
        self.files_to_download = [
            TextFileInfo(
                remote_path=os.path.join(self.remote_path, self.climatology_dir, self.climate_file),
                local_filename=self.climate_file
            ),
            TextFileInfo(
                remote_path=os.path.join(self.remote_path, self.climatology_dir, self.current_file),
                local_filename=self.current_file
            ),
            TextFileInfo(
                remote_path=os.path.join(self.remote_path, self.climatology_dir, self.previous_file),
                local_filename=self.previous_file
            )
        ]

    async def download_files(self) -> Dict:
        """Download text files from the remote server."""
        result = {
            'downloaded': 0,
            'failed': []
        }
        
        # Run SCP in a separate thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        try:
            download_result = await loop.run_in_executor(
                None, self._download_files_sync
            )
            result.update(download_result)
        except Exception as e:
            self.logger.error(f"Error downloading text files: {e}")
            for file_info in self.files_to_download:
                result['failed'].append(file_info.local_filename)
                
        return result
    
    def _download_files_sync(self) -> Dict:
        """Synchronous method to download files via SCP."""
        result = {
            'downloaded': 0,
            'failed': []
        }
        
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        try:
            # Connect to the remote server
            ssh_client.connect(
                hostname=self.ssh_host,
                username=self.ssh_user,
                key_filename=os.path.expanduser(self.ssh_key_path)
            )
            
            # Create SCP client
            with ssh_client.open_sftp() as sftp:
                for file_info in self.files_to_download:
                    local_path = self.output_dir / file_info.local_filename
                    try:
                        sftp.get(file_info.remote_path, str(local_path))
                        self.logger.info(f"Successfully downloaded {file_info.local_filename}")
                        result['downloaded'] += 1
                    except Exception as e:
                        self.logger.error(f"Failed to download {file_info.local_filename}: {e}")
                        result['failed'].append(file_info.local_filename)
                        
        except Exception as e:
            self.logger.error(f"SSH connection error: {e}")
            # Mark all files as failed if we couldn't connect
            for file_info in self.files_to_download:
                if file_info.local_filename not in result['failed']:
                    result['failed'].append(file_info.local_filename)
        finally:
            ssh_client.close()
            
        return result