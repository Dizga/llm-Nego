import json
import os
import copy
from statistics import mean
import numpy as np
from utils.export_ppo_training_set import export_ppo_training_set

def extract_ppo_dataset(
    folder_path: str,
    last_k_responses=None,
    remove_errors=False,
    score_function="points_score",
    score_function_kwargs=None,
    scores_pp_func=None,
    scores_pp_func_kwargs=None,
    filter_out_func=None,
    filter_out_func_kwargs=None,
):
    """
    Extracts data for HF PPO training from game logs and logs each conversation in a subfolder.

    Parameters:
    - folder_path (str): Path to the folder containing conversation JSON files.
    - last_k_responses (int or None): If set, only the last k assistant messages will be trained on.
    - remove_errors (bool): If True, removes messages marked as errors.
    - score_function (callable or None): Function to calculate scores for responses.
    - score_function_kwargs (dict or None): Additional keyword arguments for the score function.

    Returns:
    - queries (list): List of queries (contexts) extracted from the conversations.
    - responses (list): List of assistant responses.
    - scores (list): List of scores associated with the responses.
    """

    queries, responses, scores = [], [], []

    # Create a subfolder for exported conversations
    export_folder = os.path.join(folder_path, "ppo_conv_data")
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)

    for file_name in os.listdir(folder_path):
        # Skip the export folder itself if it exists
        if file_name == "ppo_conv_data":
            continue

        # Import conversation
        conversation_path = os.path.join(folder_path, file_name)
        with open(conversation_path, "r") as file:
            conversation = json.load(file)

        # Process conversation
        conv_queries, conv_responses, conv_scores = process_conversation(
            conversation,
            last_k_responses=last_k_responses,
            remove_errors=remove_errors,
            score_function=score_function,
            score_function_kwargs=score_function_kwargs,
        )

        queries.extend(conv_queries)
        responses.extend(conv_responses)
        scores.extend(conv_scores)

        # Export each conversation to the subfolder
        export_file_path = os.path.join(export_folder, f'ppo_{file_name}')
        export_ppo_training_set(export_file_path, conv_queries, conv_responses, conv_scores)

    if scores_pp_func is not None:
        scores = globals()[scores_pp_func](scores, **scores_pp_func_kwargs)

    if filter_out_func is not None:
        queries, responses, scores = globals()[filter_out_func](queries, responses, scores, **filter_out_func_kwargs)

    return queries, responses, scores


def process_conversation(
    conversation,
    score_function=lambda x: 10 if x["self_points"] > 0 else 0,
    score_function_kwargs={},
    last_k_responses=None,
    remove_errors=False,
):
    """
    Processes a single conversation and extracts queries, responses, and scores.

    Parameters:
    - conversation (list): List of message dictionaries representing a conversation.
    - score_function (callable): Function to calculate scores for responses.
    - score_function_kwargs (dict or None): Additional keyword arguments for the score function.
    - last_k_responses (int or None): If set, only the last k assistant messages will be trained on.
                                      If None, all messages are considered.
    - remove_errors (bool): If True, removes messages marked as errors.

    Returns:
    - conversation_queries (list): List of queries (contexts) extracted from the conversation.
    - conversation_responses (list): List of assistant responses.
    - conversation_scores (list): List of scores associated with the responses.
    """
    context = []
    conversation_queries = []
    conversation_responses = []
    conversation_scores = []

    round_nb = -1
    round_msg_nb = -1


    scores = globals()[score_function](conversation, **score_function_kwargs)

    for message in conversation:
        if message.get("is_error") and remove_errors:
            continue

        elif message.get("role") == "round_info":
            round_nb += 1
            round_msg_nb = 0
            
        elif message.get("role") == "user" or message.get("role") == "assistant":
            message = {"role": message["role"], "content": message["content"]}
            context.append(message)

        # Collect assistant responses
        if message.get("role") == "assistant":
            conversation_queries.append(copy.deepcopy(context[:-1]))
            conversation_responses.append([message])
            score = scores[round_nb]
            conversation_scores.append(score)

        round_msg_nb += 1

    # Limit to the last k assistant messages if specified
    if last_k_responses != -1:
        conversation_queries = conversation_queries[-last_k_responses:]
        conversation_responses = conversation_responses[-last_k_responses:]
        conversation_scores = conversation_scores[-last_k_responses:]

    return conversation_queries, conversation_responses, conversation_scores


def score_based_on_agreement(conversation, points_on_agreement=10):
    """
    Sets the score to `points_on_agreement` if an agreement was reached in each round, else 0.

    Parameters:
    - conversation (list): List of message dictionaries representing a conversation.
    - points_on_agreement (int): Points to assign if an agreement was reached.

    Returns:
    - list: Scores for each round based on agreement.
    """
    scores = []
    for message in conversation:
        if message.get("role") == "round_info":
            agreement_reached = message["content"]["agreement_reached"]
            score = points_on_agreement if agreement_reached else 0
            scores.append(score)
    return scores


def points_score(conversation, no_agreement_score=-1, exponent=1.0, return_discounted=True, discount_factor=0.6):
    """
    Sets the score to the points reached in each round, applying a discount factor if specified.

    Parameters:
    - conversation (list): List of message dictionaries representing a conversation.
    - no_agreement_score (int): Score to assign if no agreement is reached.
    - exponent (float): Exponent to apply to the self_points.
    - return_discounted (bool): Whether to apply discounting to the scores.
    - discount_factor (float): Discount factor to apply to future rewards.

    Returns:
    - list: Discounted scores for each round based on current round points.
    """
    scores = []
    discounted_score = 0

    for message in reversed(conversation):
        if message.get("role") == "round_info":
            agreement_reached = message["content"]["agreement_reached"]
            self_points = message["content"]["self_points"]
            score = self_points ** exponent if agreement_reached else no_agreement_score

            if return_discounted:
                discounted_score = score + discount_factor * discounted_score
            else:
                discounted_score = score

            scores.insert(0, discounted_score)

    return scores

def advantage_alignment_score(conversation, 
                 no_agreement_score=-1, 
                 exponent=1.0, 
                 return_discounted=True, 
                 discount_factor=0.6):

    scores = []
    self_returns = []
    other_returns = []

    # Compute the returns at each step for both players (traverse the rounds in reverse)
    total_self_points = 0
    total_other_points = 0

    for i in reversed(range(len(conversation))):
        message = conversation[i]
        if message.get("role") == "round_info":
            self_points = message["content"]["self_points"]
            other_points = message["content"]["other_points"]

            # Apply discount factor to the points
            total_self_points = self_points + discount_factor * total_self_points
            total_other_points = other_points + discount_factor * total_other_points

            self_returns.append(total_self_points)
            other_returns.append(total_other_points)

    # Convert lists to numpy arrays
    self_returns = np.array(self_returns)
    other_returns = np.array(other_returns)

    nb_rounds = len(self_returns)
    # Compute advantage alignment scores
    for i in range(nb_rounds):
        if i > 0:
            scores.append(float(np.sum(self_returns[:i]) * other_returns[i]))
        else:
            scores.append(float(self_returns[0] * other_returns[0]))
    return scores

def score_based_on_future_points(conversation):
    """
    Sets the score to the sum of points from the current round to the end of the game.

    Parameters:
    - conversation (list): List of message dictionaries representing a conversation.

    Returns:
    - list: Scores for each round based on future points.
    """
    scores = []
    total_points = 0
    for message in reversed(conversation):
        if message.get("role") == "round_info":
            self_points = message["content"]["self_points"]
            total_points += self_points
            scores.insert(0, total_points)
    return scores


def positive_score_filter(queries, responses, scores):
    """
    Filters out responses with non-positive scores.

    Parameters:
    - queries (list): List of queries (contexts) extracted from the conversations.
    - responses (list): List of assistant responses.
    - scores (list): List of scores associated with the responses.

    Returns:
    - filtered_queries (list): List of queries with positive scores.
    - filtered_responses (list): List of responses with positive scores.
    - filtered_scores (list): List of positive scores.
    """
    filtered_queries = []
    filtered_responses = []
    filtered_scores = []

    for query, response, score in zip(queries, responses, scores):
        if score > 0:
            filtered_queries.append(query)
            filtered_responses.append(response)
            filtered_scores.append(score)

    return filtered_queries, filtered_responses, filtered_scores


def subtract_mean_positive_score(scores, min_score=0):
    positive_scores = [s for s in scores if s > min_score]
    mean_score = mean(positive_scores)
    return [s - mean_score for s in scores if s > min_score]


def subtract_mean_score(scores):
    mean_score = mean(scores)
    return [s - mean_score for s in scores]
