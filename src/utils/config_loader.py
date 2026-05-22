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
            try:
                self.config_data = json.load(file)
            except json.JSONDecodeError as e:
                raise ValueError(f"Error decoding JSON: {e}")

        return self.config_data

    def get(self, key: str, default: Any = None) -> Any:
        return self.config_data.get(key, default)

# usage example
if __name__ == "__main__":
    # TODO: replace with your actual config file path
    config_loader = ConfigLoader("path/to/config.json")
    try:
        config = config_loader.load()
        print(f"Loaded config: {config}")
    except Exception as e:
        print(f"Failed to load config: {e}")