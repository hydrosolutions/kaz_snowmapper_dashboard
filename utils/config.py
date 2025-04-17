import yaml
import os
from pathlib import Path
from typing import Dict, Any
import logging

class ConfigLoader:
    """Handles loading and merging of configuration files."""

    def __init__(self):
        self.logger = logging.getLogger('snowmapper.config')
        self.project_root = Path(__file__).parent.parent

    def load_config(self, env: str = 'local') -> Dict[Any, Any]:
        """Load and merge configuration for specified environment."""
        # Load base config first
        base_config = self._load_yaml_file('config.base.yaml')

        # Load environment specific config
        env_config = self._load_yaml_file(f'config.{env}.yaml')

        # Merge configurations
        merged_config = self._deep_merge(base_config, env_config)

        # Replace environment variables
        merged_config = self._replace_env_vars(merged_config)

        # Overwrite ssh key path if env is AWS
        if env == 'aws':
            merged_config['ssh']['key_path'] = f"/app/processing/{merged_config['ssh']['key_path']}"

        return merged_config

    def _load_yaml_file(self, filename: str) -> Dict[str, Any]:
        """Load a YAML configuration file."""
        config_path = self.project_root / 'config' / filename
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Error loading config file {filename}: {e}")
            return None

    def _resolve_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert relative paths to absolute paths."""
        path_keys = ['input_dir', 'output_dir', 'cache_dir', 'mask_path']

        for key in path_keys:
            if key in config and not Path(config[key]).is_absolute():
                config[key] = str(self.project_root / config[key])

        return config

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Recursively merge two dictionaries."""
        merged = base.copy()

        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._deep_merge(merged[key], value)
            else:
                merged[key] = value

        return merged

    def _replace_env_vars(self, config: Dict) -> Dict:
        """Recursively replace environment variables in config values."""
        if isinstance(config, dict):
            return {key: self._replace_env_vars(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self._replace_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            env_var = config[2:-1]
            return os.getenv(env_var, config)
        return config