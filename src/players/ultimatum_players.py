import json


class UltimatumPlayer:
    def __init__(self, 
                 game_intro_file, 
                 chain_of_thought_file,
                 max_retries,
                 agent, 
                 player_type="player_0",
                 **kwargs):
        """
        Initializes the Ultimatum Player.

        Args:
            agent (NegoAgent): The LLM player instance.
            player_type (str): The type of player, either "player_0" or "player_1".
        """

        with open(game_intro_file, 'r') as file:
            self.game_basics = file.read()
        
        self.chain_of_thought = None
        if chain_of_thought_file:
            with open(chain_of_thought_file, 'r') as file:
                self.chain_of_thought = file.read()
        
        self.max_retries = max_retries

        self.agent = agent
        self.player_type = player_type
        self.is_new_game = True

    def play_move(self, state):
        """
        Plays a move in the Ultimatum game.

        Args:
            state (dict): The current state of the game.

        Returns:
            str: The player's response.
        """
        # Create the context for the agent to generate a response
        context = self.get_context(state)
        response = self.agent.prompt(context)

        # Validate and process the response
        valid_response, error_message = self.validate_response(response)
        retries = 0

        while not valid_response and retries < 3:
            response = self.agent.prompt(error_message, is_error=True)
            valid_response, error_message = self.validate_response(response)
            retries += 1

        if not valid_response:
            response = """<reason>50/50</reason><message>50/50</message><propose>{ "i_take": 50, "other_player_gets": 50 }</propose>"""

        self.is_new_game = False
        return response

    def get_context(self, state):
        """
        Constructs the context for the agent based on the current state.

        Args:
            state (dict): The current state of the game.

        Returns:
            str: The context message for the agent.
        """

        user_message = ""
        if self.is_new_game:
            user_message += self.game_basics.format(**state)

        user_message += f"The other player said: '{state['last_message']}'\n" if state['last_message'] else "You are the first to play, there is no messages yet.\n"
        user_message += f"The other player proposition is '{state['last_proposition']}'\n" if state['last_proposition'] else ""

        is_last_turn = state['turn'] == state['max_turns']

        if self.chain_of_thought is not None and not is_last_turn:
            pass
            user_message += self.chain_of_thought.format(**state)

        if state['turn'] == state['max_turns'] - 1:
            user_message += "This is the last turn. If you don't accept the last proposition or the other player doesn't accept your next proposition, both players get $0.\n"
        if state['turn'] == state['max_turns']:
            user_message += "It is the last turn, you cannot negotiate anymore, send an empty message. Using the <reason> tag, reason about the exchanges so far and if you should you accept the last proposition or refuse. If you refuse both players will receive $0."
            # user_message += "This is the last turn, no more negotiation is possible. If you don't accept the last proposition both players get $0.\n"
        return user_message


    def validate_response(self, response):
        """
        Validates the response from the agent.

        Args:
            response (str): The agent's response.

        Returns:
            tuple: A tuple containing a boolean indicating validity and an error message if invalid.
        """
        errors = []
        
        if "<reason>" not in response or "</reason>" not in response:
            errors.append("Missing <reason>...</reason> tag.")
        
        if "<message>" not in response or "</message>" not in response:
            errors.append("Missing <message>...</message> tag.")
        
        if "<propose>" not in response or "</propose>" not in response:
            errors.append("Missing <propose>...</propose> tag.")
        else:
            try:
                propose_content = response.split("<propose>")[1].split("</propose>")[0].strip()
                if propose_content != "accept":
                    proposal = json.loads(propose_content)
                    if not all(key in proposal for key in ["i_take", "other_player_gets"]):
                        errors.append('The <propose> tag must contain JSON with keys "i_take" and "other_player_gets".')
            except json.JSONDecodeError:
                errors.append("The content within <propose>...</propose> is not valid JSON.")
        
        if errors:
            return False, "Errors: " + "; ".join(errors)
        return True, "Response is valid."

    def new_round(self):
        """
        Resets round attibutes.

        """
        self.first_turn = True
    
    def new_game(self):
        """
        Resets the message history of the LLM player.

        Returns:
            list: The message history before resetting.
        """
        self.new_round()
        self.is_new_game = True
        history = self.agent.history
        self.agent.reset_messages()
        return history
    
    def get_history(self):
        return self.agent.history