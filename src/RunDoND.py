import json
import numpy as np
from prompts.instruction import get_instruction_prompt
from store import add_run_to_store
from type.behavior import Behavior
from utils import generate_initial_state
from logger import Logger
from DoND import DoND
from agents import DoNDagent
import hydra
from datetime import datetime
import os 

class TwoPlayersNego:
    def __init__(self, iterations_per_run, games_per_iteration, game, player_0, player_1, logger):
        self.iterations_per_run = iterations_per_run
        self.games_per_iteration = games_per_iteration
        self.game = game
        self.player_0 = player_0
        self.player_1 = player_1
        self.logger = logger

    def run_iterations(self):
        for _ in range(self.iterations_per_run):
            for _ in range(self.games_per_iteration):
                game_result = self.run_game()
                self.logger.log_game(game_result)
            self.logger.new_iteration()

    def run_game(self):
        quantities, values_p0, values_p1 = self.game.reset()
        self.player_0.instruct(quantities, values_p0)
        self.player_1.instruct(quantities, values_p1)
        ongoing = True
        message = None
        while ongoing:
            if self.game.current_turn() == "p0":
                message = self.player_0.play(message)
                ongoing = self.game.step(message)
            else:
                message = self.player_1.play(message)
                ongoing = self.game.step(message)
        return self.game.export()

@hydra.main(config_path="../conf", config_name="config")
def RunDoND(cfg):

    # Make output directory
    out_dir = f"DATA/RUN_{datetime.now().strftime('%Y%m%d_%H%M%S')}" 
    os.makedirs(out_dir, exist_ok=True)

    logger = Logger(out_dir)
    
    game = DoND()

    print(os.getcwd())
    with open(cfg['player_0']['instructions'], "r") as instruction_prompt_file:
        instruction_text = instruction_prompt_file.read()

    with open(cfg['player_0']['CoT'], "r") as cot_prompt_file:
        cot_text = cot_prompt_file.read()

    player_0 = DoNDagent(
        name="agent",
        device=cfg['device'],
        model=cfg['player_0']['model'],
        tokenizer=cfg['player_0']['tokenizer'],
        chain_of_thought=cot_text,
        instructions=instruction_text
    )

    with open(cfg['player_1']['instructions'], "r") as instruction_prompt_file:
        instruction_text = instruction_prompt_file.read()

    with open(cfg['player_1']['CoT'], "r") as cot_prompt_file:
        cot_text = cot_prompt_file.read()

    player_1 = DoNDagent(
        name="agent",
        device=cfg['device'],
        model=cfg['player_1']['model'],
        tokenizer=cfg['player_1']['tokenizer'],
        chain_of_thought=cot_text,
        instructions=instruction_text
    )

    run_handler = TwoPlayersNego(
        iterations_per_run=cfg["run"]["nb_iterations"],
        games_per_iteration=cfg["run"]["games_per_iterations"],
        game=game,
        player_0=player_0,
        player_1=player_1,
        logger=logger
    )

    run_handler.run_iterations()

if __name__ == "__main__":
    RunDoND()
