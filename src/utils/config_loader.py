import json
import os
from typing import Any, Dict

class ConfigLoader:
    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self.config: Dict[str, Any] = {}

    def load(self) -> None:
        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        
        with open(self.config_path, 'r') as file:
            self.config = json.load(file)

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.config[key] = value

    def save(self) -> None:
        with open(self.config_path, 'w') as file:
            json.dump(self.config, file, indent=4)

# usage example
# if __name__ == "__main__":
#     config_loader = ConfigLoader("config.json")
#     config_loader.load()
#     print(config_loader.get("some_key", "default_value"))  # for testing
#     config_loader.set("new_key", "new_value")
#     config_loader.save()  # TODO: add error handling here