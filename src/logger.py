import os
import json
import logging
import logging.config
from datetime import datetime
from omegaconf import OmegaConf

class Logger:
    def __init__(self, name, cfg):
        self.datenow = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = os.path.join(cfg.log_dir, self.datenow)
        os.makedirs(self.log_dir, exist_ok=True)

        self.setup_logging(cfg.log_config)
        self.logger = logging.getLogger(name)

        self.instructions_file = os.path.join(self.log_dir, 'instructions.txt')
        self.log_instructions(cfg.instructions)

        self.games_log_file = os.path.join(self.log_dir, 'games_log.txt')

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

    def log_instructions(self, instructions: str):
        "Log the instructions to a text file."
        with open(self.instructions_file, 'w') as f:
            f.write(instructions)

    def log_game(self, game: dict):
        "Append game to the games log file."
        game_a = game['quantities'] + " $$ " + game['a_values'] + " $$ " +  a_perspective + 'a_score'
        game_b = game['quantities'] + " " + game['a_values'] + a_perspective + 'score'
        with open(self.games_log_file, 'a') as f:
            f.write(game_str + '\n')

    def save_player_messages(self, player_name: str, messages: list):
        file_path = os.path.join(self.log_dir, f"{player_name}.json")
        with open(file_path, 'w') as f:
            json.dump(messages, f, indent=4)
