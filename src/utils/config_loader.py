import json
import os

class ConfigLoader:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        
        with open(self.config_path, 'r') as file:
            return json.load(file)

    def get(self, key: str, default=None):
        # return the value for the key or default if not found
        return self.config.get(key, default)

    def set(self, key: str, value):
        # update the config with a new value
        self.config[key] = value
        self.save_config()

    def save_config(self):
        with open(self.config_path, 'w') as file:
            json.dump(self.config, file, indent=4)

# example usage
if __name__ == "__main__":
    # TODO: change to your config file path
    config_loader = ConfigLoader('path/to/config.json')
    print(config_loader.get('some_key', 'default_value'))  # change 'some_key' as needed
    config_loader.set('new_key', 'new_value')  # example of setting a new config value