def set_discounted_returns(player_info, info, discount_factor=0.99):
    """
    Sets the discounted returns for each message in the conversation.

    Args:
        player_info (dict): Contains the chat history of the player.
        info (dict): Contains the game information including returns.
        discount_factor (float): The discount factor to apply to future returns.
    """
    # Extract the chat history and returns from the game info
    chat_history = player_info.get("chat_history", [])
    round_points = info.get("round_points", [])
    player_name = player_info.get("player_name", "")

    # Calculate discounted returns for each round
    discounted_returns = []
    cumulative_return = 0
    for i in reversed(range(len(round_points))):
        # Use the player's name to get the specific points
        role = info['round_player_roles'][i].get(player_name)
        round_value = round_points[i].get(role, 0)
        cumulative_return = round_value + discount_factor * cumulative_return
        discounted_returns.insert(0, cumulative_return)

    # Set the discounted returns for each message based on round_number
    for message in chat_history:
        if message["role"] != "user":
            round_number = message["round_nb"]
            if round_number < len(discounted_returns):
                message["return"] = discounted_returns[round_number]
    return chat_history
    