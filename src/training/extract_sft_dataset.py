import json
import os
import copy
from statistics import mean
from training.extract_ppo_dataset import extract_ppo_dataset


def extract_sft_dataset(
    folder_path: str,
    filter_function="above_mean_filter",
    filter_function_kwargs={},
    substract_mean_score=False,
    normalize_scores=(0, 1),
    last_k_responses=None,
    remove_errors=False,
    score_function=None,
    score_function_kwargs={},
):
    """
    Extracts data for HF SFT training from game logs, filtering based on a custom filter function.

    Parameters:
    - folder_path (str): Path to the folder containing conversation JSON files.
    - filter_function (callable): Function to filter examples based on score.
    - filter_function_kwargs (dict or None): Additional keyword arguments for the filter function.
    - substract_mean_score (bool): If True, subtracts the mean score from all scores.
    - normalize_scores (tuple): Tuple containing min and max values for score normalization.
    - last_k_responses (int or None): If set, only the last k assistant messages will be trained on.
                                      If None, all messages are considered.
    - remove_errors (bool): If True, removes messages marked as errors.
    - score_function (callable or None): Function to calculate scores for responses.
    - score_function_kwargs (dict or None): Additional keyword arguments for the score function.

    Returns:
    - List[dict]: List of dictionaries in the conversational format.
    """

    queries, responses, scores = extract_ppo_dataset(
        folder_path,
        substract_mean_score=substract_mean_score,
        normalize_scores=normalize_scores,
        last_k_responses=1,
        remove_errors=remove_errors,
        score_function=score_function,
        score_function_kwargs=score_function_kwargs,
    )

    # Filter based on the custom filter function
    filtered_data = []
    for query, response, score in zip(queries, responses, scores):
        if globals()[filter_function](score, scores, **filter_function_kwargs):
            system = {"role": "system", "content": "You are helpful"}
            messages =  query + response
            filtered_data.append({"messages": messages})

    return filtered_data


def above_mean_filter(score, scores, **kwargs):
    """
    Filter function for SFT dataset extraction.
    """
    return score > mean(scores)

