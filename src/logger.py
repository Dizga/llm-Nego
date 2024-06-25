import os
import json
import logging
import logging.config
from datetime import datetime

class Logger:
    def __init__(self, name, log_dir: str = 'logs', log_config: str = 'logging.conf'):
        self.datenow = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = os.path.join(log_dir, self.datenow)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.setup_logging(log_config)
        self.logger = logging.getLogger(name)

    def setup_logging(self, config_file):
        logging.config.fileConfig(config_file, defaults={'date': self.datenow})

    def log_info(self, message: str):
        self.logger.info(message)

    def log_debug(self, message: str):
        self.logger.debug(message)

    def log_warning(self, message: str):
        self.logger.warning(message)

    def log_error(self, message: str):
        self.logger.error(message)

    def save_player_messages(self, player_name: str, messages: list):
        file_path = os.path.join(self.log_dir, f"{player_name}.json")
        with open(file_path, 'w') as f:
            json.dump(messages, f, indent=4)
