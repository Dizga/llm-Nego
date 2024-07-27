import json
import numpy as np
from agents import GPTAgent, HFAgent
from game import nego_game
from prompts.instruction import get_instruction_prompt
from store import add_run_to_store
from type.behavior import Behavior
from utils import generate_initial_state, parse_context
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
p1_behavior = Behavior.BASIC
p2_behavior = Behavior.BASIC
p1_rewards = []
p2_rewards = []
p1_history = []
p2_history = []

state = {
  "type_of_items": 3,
  "item_quantities": [5,5,5],
  "utilities": {
   "player_1": [2,2,2],
   "player_2": [2,2,2]
  },
  "turns": 5
 }

# states= [state]

# states = [generate_initial_state() for _ in range(2)]

# with open("states.json", 'r') as f:
#     states = json.load(f)

# for game, state in enumerate(states, 1):

with open("selfplay.txt", "r") as file:
    # read lines into a list, stripping the newline character from each line
    lines = [line.strip() for line in file]

random_indices = np.random.randint(0, 4085, size=1)

cnts, p1_vals = parse_context(lines[random_indices[i] * 2])
_, p2_vals = parse_context(lines[random_indices[i] * 2 + 1])

with open("prompts/coop.txt", "r") as system_prompt_file:
    system_text = system_prompt_file.read()

system_text.format(book_cnt=cnts[0])

for game in range(1):
    logger.log_info(f'Game {game} started.')

    cnts, p1_vals = parse_context(lines[random_indices[game] * 2])
    _, p2_vals = parse_context(lines[random_indices[game] * 2 + 1])

    state = {
    "type_of_items": 3,
    "item_quantities": cnts,
    "utilities": {
    "player_1": p1_vals,
    "player_2": p2_vals
    },
    "turns": 5
    }

    player1.reset_messages()
    player2.reset_messages()

    player1.add_system_message(get_instruction_prompt(p1_behavior))
    player2.add_system_message(get_instruction_prompt(p2_behavior))

    logger.log_info("State:")
    logger.log_info(state)

    try:
        rewards, propositions = nego_game(state, expected_turns, player1, player2, logger)

        p1_rewards.append(rewards[player1.name])
        p2_rewards.append(rewards[player2.name])
        logger.log_info("Rewards:")
        logger.log_info(rewards)
    except:
        p1_history.append(player1.messages)
        p2_history.append(player2.messages)
        break
    else:
        p1_history.append(player1.messages)
        p2_history.append(player2.messages)

    add_run_to_store(player1.type, player2.type, player1.messages, player2.messages, rewards[player1.name], rewards[player2.name], p1_behavior, p2_behavior, propositions, 0)


logger.save_player_messages(player1.name, p1_history)
logger.save_player_messages(player2.name, p2_history)

logger.log_info("Cumulative Rewards:")
logger.log_info(f'{player1.name}: {p1_rewards}')
logger.log_info(f'{player2.name}: {p2_rewards}')