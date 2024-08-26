import json
import numpy as np
import hydra
from datetime import datetime
import os
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

# local imports
from utils.dond_logger import DondLogger
from environments.dond_game import DondGame
from environments.dond_player import DondPlayer
from agents.hf_agent import HfAgent
from agents.dummy_hf_agent import DummyHfAgent
from agents.oai_agent import OaiAgent


class DondIterationRunner:
    def __init__(self, 
                 games_per_iteration, 
                 game: DondGame, 
                 player_0: DondPlayer, 
                 player_1: DondPlayer, 
                 logger: DondLogger,
                 ):

        self.games_per_iteration = games_per_iteration
        self.game = game
        self.player_0 = player_0
        self.player_1 = player_1
        self.logger = logger

    def run_iteration(self):
        self.logger.new_iteration()
        for _ in range(self.games_per_iteration):
            self.run_game()

    def run_game(self):
        self.logger.log_info("Game started.")
        self.logger.new_game()
        players = [self.player_0, self.player_1]
        self.player_0.new_game()
        self.player_1.new_game()
        game_state = self.game.reset()
        player_id = 0
        while not game_state['game_ended']:
            if game_state['new_round']:
                self.player_0.new_round()
                self.player_1.new_round()
            is_proposal, content = players[player_id].play_move(game_state)
            game_state = self.game.step(content, is_proposal=is_proposal)
            player_id = (player_id + 1) % 2
            
        # while True:
        self.logger.log_game(*self.game.export(), 
                             self.player_0.get_history(), 
                             self.player_1.get_history())
        self.logger.log_info("Game completed.")