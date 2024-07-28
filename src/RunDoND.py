import json
import numpy as np
from prompts.instruction import get_instruction_prompt
from store import add_run_to_store
from type.behavior import Behavior
from utils import generate_initial_state

from logger import *
from DoND import *
from agents import *


class TwoPlayersNego:
    def __init__(self, 
            iterations_per_run,
            games_per_iteration,
            game,
            player_0,
            player_1):
        self.iterations_per_run = iterations_per_run
        self.games_per_it = games_per_iteration
        self.game = game
        self.player_0 = player_0
        self.player_1 = player_1
        self.logger = Logger()

    def run_iterations(self):
        for k in range(self.iterations_per_run):
            for i in range(self.games_per_it):
                game = self.run_game()
                self.logger.log_game(game)
            self.logger.new_iteration()

    def run_game(self):
        "Play a game."
        self.game.reset()
        ongoing = True
        while ongoing:
            if self.game.current_turn() == "p0":
                message = self.p0.play(message)
                ongoing = self.game.step(message)
            else:
                ongoing = self.p0.play(message)
                ongoing = self.game.step(message)
        

@hydra.main(config_path="../conf", config_name="config")
def RunDoND(cfg):

    game = DoND()

    player_0 = DoNDagent(
        name = "agent",
        device = cfg['device'], 
        model = cfg['player_1']['model'],
        tokenizer = cfg['player_1']['tokenizer'],
        chain_of_thought = cfg['player_1']['chain_of_thought'],
        instructions = cfg['player_1']['instructions']
    )

    player_1 = DoNDagent(
        name = "agent",
        device = cfg['device'], 
        model = cfg['player_1']['model'],
        tokenizer = cfg['player_1']['tokenizer'],
        chain_of_thought = cfg['player_1']['chain_of_thought'],
        instructions = cfg['player_1']['instructions']
    )

    run_handler = TwoPlayersNego(iterations_per_run=cfg["iterations_per_run"],
                             games_per_iteration=cfg["games_per_iteration"],
                             game = game,
                             player_0=player_0,
                             player_1=player_1
                             )
    
    run_handler.run_iterations()

if __name__ == "__main__":
    RunDoND()
