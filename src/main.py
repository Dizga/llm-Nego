import json
import numpy as np
from agents import GPTAgent, HFAgent
from game import nego_game
from prompts.instruction import get_instruction_prompt
from store import add_run_to_store
from type.behavior import Behavior
from utils import generate_initial_state
from logger import Logger

class Master:
    def __init__(
        cfg:DictConfig
    ):
    nb_games = cfg['games_per_iteration']
    nb_iterations = cfg['iterations_per_run']
    self.game = DoND()
    self.p0 = TODO
    self.p1 = TODO

    def run_iterations(self):
        it_n = 1
        game_n = 1
        for k in range(self.nb_iterations):
            for i in range(self.games_per_it):
                game = self.run_game()
                self.logger.log_game(game, it_n, game_n)
                game_n += 1
            it_n+=1

    def run_game(self):
        "Play a game."
        result = True
        response = None
        while ongoing:
            if self.game.current_turn() == "p0":
                response = self.p0.play(response)
                result = self.game.step(response)
            else:
                response = self.p1.play(response)
                result = self.game.step(response)

@hydra.main(config_path="../conf", config_name="config")
def main():
    pass

if __name__ == "__main__":
    main()
