import json
import os
from typing import Any, Dict

class ConfigLoader:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config_data: Dict[str, Any] = {}

    def load(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        
        with open(self.config_path, 'r') as file:
            self.config_data = json.load(file)
        
        return self.config_data

    def get(self, key: str, default: Any = None) -> Any:
        return self.config_data.get(key, default)

# usage example
if __name__ == "__main__":
    config_loader = ConfigLoader("config.json")  # TODO: update with your config path
    try:
        config = config_loader.load()
        print(f"Loaded config: {config}")
    except Exception as e:
        print(f"Error loading config: {e}")