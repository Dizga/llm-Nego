import json

def gather_dond_statistics(player_info, info):
    """
    Gathers all statistics of a game for a single player and outputs them in JSONL format.

    Args:
        player_info (dict): A dictionary containing player information.
        info (dict): A dictionary containing game information.

    Returns:
        str: A JSONL string containing the game statistics.
    """
    statistics = []
    player_name = player_info['player_name']

    for i, state in enumerate(info['round_player_roles']):
        player_role = state.get(player_name)

        if player_role is None:
            continue

        other_role = next(role for role in state.values() if role != player_role)

        round_info = {
            "agreement_reached": info['round_agreements_reached'][i],
            "agreement_percentage": 100 if info['round_agreements_reached'][i] else 0,
            "self_points": info['round_points'][i][player_role],
            "other_points": info['round_points'][i][other_role],
            "points_difference": info['round_points'][i][player_role] - info['round_points'][i][other_role],
            "imbalance": calculate_imbalance(info['round_points'][i], player_role, other_role),
            "items_given_to_self": calculate_items_given_to_self(info['round_finalizations'][i][player_role]),
            "self_points_on_agreement": info['round_points'][i][player_role] if info['round_agreements_reached'][i] else None,
            "other_points_on_agreement": info['round_points'][i][other_role] if info['round_agreements_reached'][i] else None,
            "points_diff_on_agreement": (info['round_points'][i][player_role] - info['round_points'][i][other_role]) if info['round_agreements_reached'][i] else None,
            "quantities": info['round_quantities'][i],
            "values": info['round_values'][i][player_role],
        }
        statistics.append(round_info)

    return statistics

def calculate_imbalance(points, player_role, other_role):
    """
    Calculates the imbalance between the points of the player and the other player.

    Args:
        points (dict): A dictionary containing points for each role.
        player_role (str): The role of the player.
        other_role (str): The role of the other player.

    Returns:
        float: The calculated imbalance.
    """
    total_points = points[player_role] + points[other_role]
    if total_points == 0:
        return 0
    return abs((points[player_role] - points[other_role]) / total_points)

def calculate_items_given_to_self(finalization):
    if all(isinstance(x, (int, float)) for x in finalization.values()):
        return sum(finalization.values())
    return None