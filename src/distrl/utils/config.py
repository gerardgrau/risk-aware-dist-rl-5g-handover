from typing import Any
import yaml
import os

def load_config(config_path: str = "configs/config.yaml") -> dict[str, Any]:
    """
    Loads configuration from a YAML file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class Config:
    """
    Static class to hold configuration.
    """
    _config: dict[str, Any] | None = None
    _config_path: str = "configs/config.yaml"

    @classmethod
    def set_config_path(cls, path: str) -> None:
        cls._config_path = path
        cls._config = None # Reset to force reload

    @classmethod
    def get(cls, key: str | None = None, default: Any = None) -> Any:
        if cls._config is None:
            cls._config = load_config(cls._config_path)
        
        if key is None:
            return cls._config
        
        return cls._config.get(key, default)
