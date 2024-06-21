import random
import json
import numpy as np
from scipy.stats import poisson
from agents import HFAgent, GPTAgent


def generate_initial_state(items_min=3, items_max=3, quantity_min=1, quantity_max=5, utility_min=1, utility_max=5, player_1_name = 'player_1', player_2_name = 'player_2'):
    type_of_items = random.randint(items_min, items_max)
    item_quantities = [random.randint(quantity_min, quantity_max) for _ in range(type_of_items)]
    player_1_utility_values = [random.randint(utility_min, utility_max) for _ in range(type_of_items)]
    player_2_utility_values = [random.randint(utility_min, utility_max) for _ in range(type_of_items)]
    return {
        "type_of_items": type_of_items,
        "item_quantities": item_quantities,
        "utilities":{
            player_1_name: player_1_utility_values,
            player_2_name: player_2_utility_values
        }
    }

def calculate_remaining_items(item_quantities, proposal):
    return [item_quantities[i] - proposal[i] for i in range(len(item_quantities))]

def generate_prompt(turn, expected_turns, state, player_turn, opponent_turn, opponent_proposal=None):

    type_of_items, item_quantities, utilities= state.values()

    player_utilities = utilities[player_turn]
    opponent_utilities = utilities[opponent_turn]

    item_info = ", ".join([f"{item_quantities[i]} units of item_{i+1}" for i in range(len(item_quantities))])
    your_utilities = ", ".join([f"{player_utilities[i]} for item_{i+1}" for i in range(len(player_utilities))])
    opponent_utilities = ", ".join([f"{opponent_utilities[i]} for item_{i+1}" for i in range(len(opponent_utilities))])

    game_ending_prob = calculate_end_probability(turn, expected_turns)

    if opponent_proposal is None:
        opponent_proposal_text = "There is no opponent proposal yet. Make a proposal."
    else:
        remaining_items = calculate_remaining_items(item_quantities, opponent_proposal)
        remaining_items_text = ", ".join([f"{remaining_items[i]} of item_{i+1}" for i in range(len(remaining_items))])
        opponent_proposal_text = (
            f"The opponent's proposal is to take "
            + ", ".join([f"{opponent_proposal[i]} of item_{i+1}" for i in range(len(opponent_proposal))]) +
            f" and leave you with {remaining_items_text}."
            "Reason about the current state of the game, then choose to accept or decline the proposal. "
            "If you decline this proposal, reason about a new proposal. "
            "If you decide that the best option is to accept your opponent's proposal, clearly say it at the end of your reasoning."
        )

    return (
        f"It is Turn {turn}. The probability of the game ending is {game_ending_prob:0.2f}. There are {item_info}. "
        f"Your utility values for the items are: {your_utilities}. "
        f"The opponent's utility values for the items are: {opponent_utilities}. "
        f"{opponent_proposal_text} "
    )

def calculate_end_probability(turn, lambda_):
    P_T_eq_t = poisson.pmf(turn, lambda_)
    P_T_ge_t = 1 - poisson.cdf(turn - 1, lambda_)
    
    if P_T_ge_t == 0:
        return 1.0
    
    P_end_at_t_given_reached_t = P_T_eq_t / P_T_ge_t
    return P_end_at_t_given_reached_t


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

def generate_json_prompt(types_of_items):
    item_info = ", ".join(["int" for i in range(types_of_items)])

    proposal_template = f"""{{
        "accept_opponent_proposal": true | false,
        "my_proposal": null | Tuple[{item_info}]
    }}"""
    return (f"Return your answer as a valid JSON string following this template: {proposal_template}, 'my_proposal' should be a list of length {types_of_items}. "
            "No explanation needed. No Markdown needed")


def calculate_rewards(state, player, opponent, proposal):

    type_of_items, item_quantities, utilities= state.values()

    player_utilities = utilities[player]
    opponent_utilities = utilities[opponent]

    remaining_items = calculate_remaining_items(item_quantities, proposal)

    return {
        player: sum([remaining_items[i] * player_utilities[i] for i in range(type_of_items)]),
        opponent:sum([proposal[i] * opponent_utilities[i] for i in range(type_of_items)])
    }

def nego_game(state, turns, expected_turns, player1, player2):
    current_proposal = None
    max_retries = 3

    for turn in range(turns):

        print(f"Turn {turn}.")

        for player, opponent in [(player1, player2), (player2, player1)]:

            reasoning_prompt = generate_prompt(turn, expected_turns, state, player.name, opponent.name, current_proposal)
            player.add_message('user', reasoning_prompt)
            player()
            produce_json_prompt = generate_json_prompt(state['type_of_items'])
            player.add_message('user', produce_json_prompt)

            retries = 0
            while retries < max_retries:
                try:
                    player_response = player()
                    player_proposal = json.loads(player_response)
                    if player_proposal["my_proposal"] is not None and len(player_proposal["my_proposal"]) != state['type_of_items']:
                        retries += 1
                        print(f"Error in proposal from {player.name} response. Retry {retries} of {max_retries}.")
                        player.add_message("user", "Invalid proposal. Your proposal should be a list of length {types_of_items}. Please try again.")
                    else:
                        break
                except json.JSONDecodeError:
                    retries += 1
                    print(f"Error decoding JSON from {player.name} response. Retry {retries} of {max_retries}.")
                    player.add_message("user", "Invalid response. Please try again and return a valid JSON string.")

            if retries == max_retries:
                print(f"Error decoding JSON from {player.name}.")
                raise

            if player_proposal["accept_opponent_proposal"]:
                print("Game ended with acceptance.")
                print(current_proposal)
                return calculate_rewards(state, player.name, opponent.name, current_proposal)

            current_proposal = player_proposal["my_proposal"]

            print(f"{player.name} proposal {current_proposal}.")

    return {
        player1.name: 0,
        player2.name: 0,
    }


player1 = HFAgent("player_1")
player2 = HFAgent("player_2")

# player1 = GPTAgent("player_1")
# player2 = GPTAgent("player_2")

player1.add_system_message(instruction_prompt)
player2.add_system_message(instruction_prompt)

state = generate_initial_state()
print("State:")
print(state)

expected_turns = 5
num_turns = np.random.poisson(expected_turns)
print(f"There will be {num_turns} turns.")

rewards = nego_game(state, num_turns, expected_turns, player1, player2)

print("Rewards:")
print(rewards)