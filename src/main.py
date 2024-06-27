import json
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


player1 = GPTAgent('player_1')
player2 = GPTAgent("player_2")
# player1 = HFAgent('player_1')
# player2 = HFAgent('player_2')
expected_turns = 5
p1_rewards = []
p2_rewards = []
p1_history = []
p2_history = []

# states = [generate_initial_state() for _ in range(2)]

with open("states.json", 'r') as f:
    states = json.load(f)

for game, state in enumerate(states, 1):
    logger.log_info(f'Game {game} started.')

    player1.reset_messages()
    player2.reset_messages()

    player1.add_system_message(instruction_prompt)
    player2.add_system_message(instruction_prompt)

    logger.log_info("State:")
    logger.log_info(state)

    rewards = nego_game(state, expected_turns, player1, player2, logger)

    p1_rewards.append(rewards[player1.name])
    p2_rewards.append(rewards[player2.name])
    p1_history.append(player1.messages)
    p2_history.append(player2.messages)

    logger.log_info("Rewards:")
    logger.log_info(rewards)

logger.save_player_messages(player1.name, p1_history)
logger.save_player_messages(player2.name, p2_history)

logger.log_info("Cumulative Rewards:")
logger.log_info(f'{player1.name}: {p1_rewards}')
logger.log_info(f'{player2.name}: {p2_rewards}')