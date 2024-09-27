import json
import os
import copy
from statistics import mean


def extract_ppo_dataset(
    folder_path: str,
    substract_mean_score=False,
    normalize_scores=(0, 1),
    last_k_responses=None,
    remove_errors=False,
    score_function=None,
    score_function_kwargs=None,
):
    """
    Extracts data for HF PPO training from game logs.

    Parameters:
    - folder_path (str): Path to the folder containing conversation JSON files.
    - substract_mean_score (bool): If True, subtracts the mean score from all scores.
    - normalize_scores (tuple): Tuple containing min and max values for score normalization.
    - last_k_responses (int or None): If set, only the last k assistant messages will be trained on.
                                      If None, all messages are considered.
    - remove_errors (bool): If True, removes messages marked as errors.
    - score_function (callable or None): Function to calculate scores for responses.
    - score_function_kwargs (dict or None): Additional keyword arguments for the score function.

    Returns:
    - queries (list): List of queries (contexts) extracted from the conversations.
    - responses (list): List of assistant responses.
    - scores (list): List of scores associated with the responses.
    """

    queries, responses, scores = [], [], []

    for file_name in os.listdir(folder_path):
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

    # Adjust scores by subtracting the mean
    if substract_mean_score and scores:
        mean_score = mean(scores)
        scores = [s - mean_score for s in scores]

    # Normalize scores
    if normalize_scores:
        min_score, max_score = normalize_scores
        scores = [(s - min_score) / (max_score - min_score) for s in scores]

    return queries, responses, scores


def process_conversation(
    conversation,
    score_function=lambda x: 10 if x["self_points"] > 0 else 0,
    score_function_kwargs=None,
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

    round_agreements = []
    round_self_points = []
    round_opponent_points = []
    round_nb = -1
    round_msg_nb = 0

    for message in conversation:
        if message.get("role") == "round_information":
            round_agreements.append(message["agreement_reached"])
            round_self_points.append(message["self_points"])
            round_opponent_points.append(message["opponent_points"])

    score_info = {
        "round_agreements": round_agreements,
        "round_self_points": round_self_points,
        "round_opponent_points": round_opponent_points,
        "round_nb": round_nb,
        "round_msg_nb": round_msg_nb,
    }

    for message in conversation:
        if message.get("is_error") and remove_errors:
            continue
        elif message.get("role") == "round_information":
            score_info["round_nb"] = round_nb
            score_info["round_msg_nb"] = round_msg_nb
        elif message.get("role") == "user" or message.get("role") == "assistant":
            context.append(message)

        # Collect assistant responses
        if message.get("role") == "assistant":
            conversation_queries.append(copy.deepcopy(context[:-1]))
            conversation_responses.append([message])
            score = score_function(score_info, **score_function_kwargs)
            conversation_scores.append(score)

        round_msg_nb += 1

    # Limit to the last k assistant messages if specified
    if last_k_responses is not None:
        conversation_queries = conversation_queries[-last_k_responses:]
        conversation_responses = conversation_responses[-last_k_responses:]
        conversation_scores = conversation_scores[-last_k_responses:]

    return conversation_queries, conversation_responses, conversation_scores


def score_based_on_agreement(score_info, points_on_agreement=10):
    """
    Sets the score to `points_on_agreement` if an agreement was reached in the current round, else 0.

    Parameters:
    - score_info (dict): Information about the current round and points.
    - points_on_agreement (int): Points to assign if an agreement was reached.

    Returns:
    - int: Score based on agreement.
    """
    current_round = score_info["round_nb"]
    if current_round >= 0 and score_info["round_agreements"][current_round]:
        return points_on_agreement
    return 0


def score_based_on_current_round_points(score_info):
    """
    Sets the score to the points reached in the current round.

    Parameters:
    - score_info (dict): Information about the current round and points.

    Returns:
    - int: Score based on current round points.
    """
    current_round = score_info["round_nb"]
    if current_round >= 0:
        return score_info["round_self_points"][current_round]
    return 0


def score_based_on_future_points(score_info):
    """
    Sets the score to the sum of points from now to the end of the game.

    Parameters:
    - score_info (dict): Information about the current round and points.

    Returns:
    - int: Score based on future points.
    """
    current_round = score_info["round_nb"]
    if current_round >= 0:
        return sum(score_info["round_self_points"][current_round:])
    return 0