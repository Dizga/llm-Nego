def game_over_condition(game_state, **kwargs):
    """
    Checks if the game is over based on the game state.

    Args:
        game_state (dict): The current state of the game.

    Returns:
        bool: True if the game is over, False otherwise.
    """
    return game_state.get("game_over", False)

def final_proposal_condition(game_state, **kwargs):
    """
    Checks if at least one player has made a final proposal.

    Args:
        game_state (dict): The current state of the game.

    Returns:
        bool: True if at least one player has made a final proposal, False otherwise.
    """
    return any(game_state.get("role_props", {}).get(role) for role in game_state.get("roles", []))
