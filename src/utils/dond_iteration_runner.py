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
                 players: list[DondPlayer],
                 logger: DondLogger,
                 ):

        self.games_per_iteration = games_per_iteration
        self.game = game
        self.players = players
        self.logger = logger

    def run_iteration(self):
        self.logger.new_iteration()
        for _ in range(self.games_per_iteration):
            self.run_game()

    def run_game(self):
        self.logger.log_info("Game started.")
        self.logger.new_game()
        self._start_new_game()
        game_state = self.game.reset()
        player_id = 0
        while not game_state['game_ended']:
            if game_state['new_round']:
                self._start_new_round()
            is_proposal, content = self.players[player_id].play_move(game_state)
            game_state = self.game.step(content, is_proposal=is_proposal)
            player_id = (player_id + 1) % 2
            
        # while True:
        self.logger.log_game(*self.game.export(), 
                             self.players[0].get_history(), 
                             self.players[1].get_history())
        self.logger.log_info("Game completed.")

    def _start_new_game(self):
        for player in self.players:
            player.new_game()

    def _start_new_round(self):
        for player in self.players:
            player.new_round()