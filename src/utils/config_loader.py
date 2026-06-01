import json
import os

class ConfigLoader:
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.config_data = {}
        self.load_config()

    def load_config(self) -> None:
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f'Config file not found: {self.config_file}')
        with open(self.config_file, 'r') as file:
            self.config_data = json.load(file)

    def get(self, key: str, default=None):
        return self.config_data.get(key, default)

    def set(self, key: str, value) -> None:
        self.config_data[key] = value
        self.save_config()

    def save_config(self) -> None:
        with open(self.config_file, 'w') as file:
            json.dump(self.config_data, file, indent=4)

# example usage
if __name__ == '__main__':
    config_loader = ConfigLoader('config.json')
    print(config_loader.get('some_setting', 'default_value'))  # TODO: replace with actual key
    config_loader.set('new_setting', 'new_value')  # TODO: update for actual usage