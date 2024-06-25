import numpy as np
from agents import GPTAgent, HFAgent
from game import nego_game
from utils import generate_initial_state
from logger import Logger

logger = Logger(__name__)

instruction_prompt = (
    "You are a player in a two-player negotiation game. "
    "Your goal is to maximize your own reward by proposing how to divide a set of items. "
    "Your reward is equal to the number of items you receive multiplied by the utility for each. "
    "You and your opponent will take turns giving proposals of how to divide the items. "
    "Each proposal specifies how many of each item it wants, leaving the remaining items for the other player.\n"
    "Before giving a proposal, each player can choose to accept the opponent's last proposal and end the game, "
    "the items would then be divided according to the accepted proposal. "
    "If no proposal is accepted after a random amount of turns, the game ends with both players receiving a reward of 0.\n"
)

player1 = GPTAgent("player_1")
# player2 = GPTAgent("player_2")
player2 = HFAgent("player_2")

player1.add_system_message(instruction_prompt)
player2.add_system_message(instruction_prompt)

state = generate_initial_state()
logger.log_info("State:")
logger.log_info(state)

expected_turns = 5
num_turns = np.random.poisson(expected_turns)
logger.log_info(f"There will be {num_turns} turns.")

rewards = nego_game(state, num_turns, expected_turns, player1, player2, logger)

logger.save_player_messages("player_1", player1.messages)
logger.save_player_messages("player_2", player2.messages)

logger.log_info("Rewards:")
logger.log_info(rewards)
