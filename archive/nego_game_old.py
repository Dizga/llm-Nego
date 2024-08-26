import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from openai import OpenAI
import json
import numpy as np
import os

# Load the model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

openai = OpenAI()

def generate_initial_state(items_min=3, items_max=3, quantity_min=1, quantity_max=5, utility_min=1, utility_max=5):
    type_of_items = random.randint(items_min, items_max)
    item_quantities = [random.randint(quantity_min, quantity_max) for _ in range(type_of_items)]
    your_utility_values = [random.randint(utility_min, utility_max) for _ in range(type_of_items)]
    opponent_utility_values = [random.randint(utility_min, utility_max) for _ in range(type_of_items)]
    return {
        "type_of_items": type_of_items,
        "item_quantities": item_quantities,
        "player1_utility_values": your_utility_values,
        "player2_utility_values": opponent_utility_values,
        "opponent_proposal": None
    }

def calculate_remaining_items(item_quantities, opponent_take):
    return [item_quantities[i] - opponent_take[i] for i in range(len(item_quantities))]

def generate_prompt(turn, item_quantities, your_utility_values, opponent_utility_values, opponent_proposal=None):
    item_info = ", ".join([f"{item_quantities[i]} units of item_{i+1}" for i in range(len(item_quantities))])
    your_utilities = ", ".join([f"{your_utility_values[i]} for item_{i+1}" for i in range(len(your_utility_values))])
    opponent_utilities = ", ".join([f"{opponent_utility_values[i]} for item_{i+1}" for i in range(len(opponent_utility_values))])

    if opponent_proposal is None:
        opponent_proposal_text = "There is no opponent proposal yet. Make a proposal."
    else:
        opponent_take = opponent_proposal
        remaining_items = calculate_remaining_items(item_quantities, opponent_take)
        remaining_items_text = ", ".join([f"{remaining_items[i]} of item_{i+1}" for i in range(len(remaining_items))])
        opponent_proposal_text = (
            f"The opponent's proposal is to take {opponent_take[0]} of item_1, "
            + ", ".join([f"{opponent_take[i]} of item_{i+1}" for i in range(1, len(opponent_take))]) +
            f" and leave you with {remaining_items_text}."
            "Reason about the current state of the game, then choose to accept or decline the proposal. "
            "If you decline this proposal, reason about a new proposal. "
            "If you decide that the best option is to accept your opponent's proposal, clearly say it at the end of your reasoning."
        )

    return (
        f"It is Turn {turn}. There are {item_info}. "
        f"Your utility values for the items are: {your_utilities}. "
        f"The opponent's utility values for the items are: {opponent_utilities}. "
        f"{opponent_proposal_text} "
    )

# Generate initial game state
state = generate_initial_state()

state = {
  "type_of_items": 3,
  "item_quantities": [3, 5, 4],
  "player1_utility_values": [3, 5, 1],
  "player2_utility_values": [2, 3, 3],
  "opponent_proposal": None
}
num_turns = np.random.poisson(5)
print(f"There will be {num_turns} turns.")

instruction_prompt = (
    "You are a player in a two-player negotiation game. "
    "Your goal is to maximize your own reward by proposing how to divide a set of items. "
    "Your reward is equal to the number of items you receive multiplied by the utility for each. "
    "You and your opponent will take turns giving proposals of how to divide the items. "
    "Each proposal specifies how many of each item it wants, leaving the remaining items for the other player.\n"
    "Before giving a proposal, each player can choose to accept the opponent's last proposal and end the game, "
    "the items would then be divided according to the accepted proposal. "
    "If no proposal is accepted after a random amount of turns sampled from a Poisson distribution with an expectation of 5, the game ends with both players receiving a reward of 0.\n"
)

proposal_template = """{
    "accept_opponent_proposal": true | false,
    "my_proposal": null | [int]
}"""
prompt_parsable_output = (f"Return your answer as a valid JSON string following this template: {proposal_template}, 'my_proposal' should be a list of length {state['type_of_items']}. "
                          "No explanation needed. No Markdown needed")

def chat_with_local_llm(messages):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=1000)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

def chat_with_gpt3(messages):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        max_tokens=500,
        messages=messages,
    )
    return response.choices[0].message.content

player_1_messages = [{"role": "user", "content": instruction_prompt}]
p2_messages = [{"role": "system", "content": instruction_prompt}]
messages = {
    1: [{"role": "user", "content": instruction_prompt}],
    2: [{"role": "system", "content": instruction_prompt}]
}
chat = {
    1: chat_with_local_llm,
    2: chat_with_gpt3
}

utility_values = {
    1: state["player1_utility_values"],
    2: state["player2_utility_values"]
}

current_player = 1
current_proposal = None

# for turn in range(1, num_turns + 1):
#     for current_player in [1,2]:
#         prompt = generate_prompt(turn, state["item_quantities"], utility_values[current_player], utility_values[current_player%2+1], current_proposal)
#         messages[current_player].append({"role": "user", "content": prompt}) 

#         player_response = chat[current_player](messages[current_player])
#         messages[current_player].append({"role": "assistant", "content": player_response})
#         messages[current_player].append({"role": "user", "content": prompt_parsable_output})

#         player_json_response = chat[current_player](messages[current_player])

#         try:
#             player_proposal = json.loads(player_json_response)
#         except json.JSONDecodeError:
#             print(f"Error decoding JSON from Player {current_player} response.")
#             print(player_json_response)
#             break

#         if player_proposal["accept_opponent_proposal"]:
#             print("Game ended with acceptance.")
#             print(current_proposal)
#             break

#         current_proposal = player_proposal["my_proposal"]

#         print(f"Player {current_player} Proposal: {current_proposal}")

# print("Game finished.")

max_retries = 3
player1_rewards, player2_rewards = 0, 0
current_proposal = None

for turn in range(1, num_turns + 1):
    player_2 = False
    prompt = generate_prompt(turn, state["item_quantities"], state["player1_utility_values"], state["player2_utility_values"], state["opponent_proposal"])
    player_1_messages.append({"role": "user", "content": prompt}) 

    player_1_response = chat_with_local_llm(player_1_messages)
    player_1_messages.append({"role": "assistant", "content": player_1_response})
    player_1_messages.append({"role": "user", "content": prompt_parsable_output})

    retries = 0
    while retries < max_retries:
        player_1_response_json = chat_with_local_llm(player_1_messages)
        player_1_messages.append({"role": "assistant", "content": player_1_response_json})

        try:
            player_1_proposal = json.loads(player_1_response_json)
            break
        except json.JSONDecodeError:
            retries += 1
            print(f"Error decoding JSON from Player {current_player} response. Retry {retries} of {max_retries}.")
            player_1_messages.append({"role": "user", "content": "Invalid response. Please try again and return a valid JSON string."})
    
    if retries == max_retries:
        print("Error decoding JSON from local LLM response.")
        print(player_1_response_json)
        break

    if player_1_proposal["accept_opponent_proposal"]:
        current_proposal = p2_proposal
        print("Game ended with acceptance.")
        print(p2_proposal)
        break

    player_2 = True
    prompt = generate_prompt(turn, state["item_quantities"], state["player2_utility_values"], state["player1_utility_values"], player_1_proposal["my_proposal"])
    p2_messages.append({"role": "user", "content": prompt})

    chatgpt_response = chat_with_gpt3(p2_messages)
    p2_messages.append({"role": "assistant", "content": chatgpt_response})
    p2_messages.append({"role": "user", "content": prompt_parsable_output})

    p2_response_json = chat_with_gpt3(p2_messages)

    try:
        p2_proposal = json.loads(p2_response_json)
    except json.JSONDecodeError:
        print("Error decoding JSON from GPT-3 response.")
        print(p2_response_json)
        break

    if p2_proposal["accept_opponent_proposal"]:
        current_proposal = player_1_proposal
        print("Game ended with acceptance.")
        print(player_1_proposal)
        break

    state["opponent_proposal"] = p2_proposal["my_proposal"]

    print(f"Turn {turn} ended.")
    print(f"Local Model Response: {player_1_proposal}")
    print(f"ChatGPT Response: {p2_proposal}")

print("Game finished.")

def calculate_rewards(item_quantities, player_1_utility_values, player_2_utility_values, proposal, player_2 = False ):

    player_1_take = proposal
    player_2_take = calculate_remaining_items(item_quantities, player_1_take)
    if player_2:
        player_1_take, player_2_take = player_2_take, player_1_take

    player1_rewards = 0
    player2_rewards = 0

    for i in range(len(item_quantities)):

        player1_rewards += player_1_take[i] * player_1_utility_values[i]
        player2_rewards += player_2_take[i] * player_2_utility_values[i]

    return player1_rewards, player2_rewards

if turn != num_turns:
    player1_rewards, player2_rewards = calculate_rewards(state["item_quantities"], state["player1_utility_values"], state["player2_utility_values"], current_proposal, player_2)

print(f'Player 1 reward: {player1_rewards}')
print(f'Player 2 reward: {player2_rewards}')