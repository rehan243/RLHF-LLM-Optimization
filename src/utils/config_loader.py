import json
import os
from typing import Any, Dict

class ConfigLoader:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config_data = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        # check if the file exists
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as file:
            try:
                # loading json data
                config = json.load(file)
                return config
            except json.JSONDecodeError as e:
                raise ValueError(f"Error parsing config file: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        # get value by key, return default if not found
        return self.config_data.get(key, default)

# example usage
if __name__ == "__main__":
    config_loader = ConfigLoader('config.json')
    print(config_loader.get('some_setting', 'default_value'))  # TODO: replace with actual config keys