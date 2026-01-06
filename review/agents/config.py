import yaml
import os
from typing import Any, Dict

class ConfigLoader:
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        # Assuming config.yml is in the root directory of the project
        # Get the directory of the current file (agents/config.py)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to the root
        root_dir = os.path.dirname(current_dir)
        config_path = os.path.join(root_dir, 'config.yml')
        
        try:
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        except FileNotFoundError:
            # Fallback or raise error
            print(f"Warning: config.yml not found at {config_path}. Using defaults.")
            self._config = {}

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default

config = ConfigLoader()
