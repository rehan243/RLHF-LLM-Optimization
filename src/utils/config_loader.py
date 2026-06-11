import json
import os

class ConfigLoader:
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.config_data = {}
        self.load_config()

    def load_config(self):
        # loading the json config file
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Config file not found: {self.config_file}")

        with open(self.config_file, 'r') as f:
            self.config_data = json.load(f)

    def get(self, key: str, default=None):
        # get a config value by key, return default if not found
        return self.config_data.get(key, default)

    def __repr__(self):
        return f"<ConfigLoader config_file={self.config_file}>"

# example usage
if __name__ == "__main__":
    config_loader = ConfigLoader('config.json')
    print(config_loader.get('some_key', 'default_value'))  # TODO: replace 'some_key' with actual key