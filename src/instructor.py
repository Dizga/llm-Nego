import json
import regex as re

from DoND import DoND
from agents import NegoAgent

class Instructor:
    """
    The Instructor acts as a middle-man between the game and a LLM player.
    """
    def __init__(self):
        """
        Initializes the Instructor.
        """
        pass

class DoNDInstructor(Instructor):
    def __init__(self, game_intro_file, chain_of_thought_file, proposal_file, dond_game:DoND, dond_player:NegoAgent, player_type="p0"):
        """
        Initializes the DoNDInstructor.

        Args:
            game_intro_file (str): Path to the file containing game introduction.
            chain_of_thought_file (str): Path to the file containing chain of thought instructions.
            dond_game (DoND): The DoND game instance.
            dond_player (NegoAgent): The LLM player instance.
            player_type (str): The type of player, either "p0" or "p1".
        """
        self.first_turn = True
        
        with open(game_intro_file, 'r') as file:
            self.game_basics = file.read()

        with open(proposal_file, 'r') as file:
            self.proposal_prompt = file.read()
        
        self.chain_of_thought = None
        if chain_of_thought_file:
            with open(chain_of_thought_file, 'r') as file:
                self.chain_of_thought = file.read()
        
        self.dond_game = dond_game
        self.dond_player = dond_player
        self.player_type = player_type

    def play_move(self):
        """
        Plays a move in the DoND game.

        Returns:
            bool: Whether the game should continue or not.
        """
        state = self.dond_game.get_state()
        if state is None:
            return False
        
        user_message = self.get_usr_message(state)
        response = self.dond_player.prompt(user_message)

        valid_response, error_message = self.validate(response)

        max_retries = float('3')
        retries = 0
        while retries < max_retries:
            if valid_response:
                break
            response = self.dond_player.prompt(error_message)
            valid_response, error_message = self.validate(response)
            retries += 1

        if retries == max_retries:
            response="<message></message>"
            #raise ValueError(f"Error validating output after {max_retries} retries.")

        is_proposal, content = self.extract(response)

        ongoing = self.dond_game.step(content, is_proposal)  # Whether the game is finished or not
        self.first_turn = False
        return ongoing

    def verificator(self, message):
        """
        Verifies if the message is correct.

        Args:
            message (str): The message to verify.

        Returns:
            bool: Always returns True (placeholder for future implementation).
        """
        # TODO: add conditions that return false if message not correct
        return message

    def get_usr_message(self, state):
        """
        Constructs a user message based on the current game state.

        Args:
            state (dict): The current state of the game.

        Returns:
            str: The constructed user message.
        """
        user_message = ""
        if self.first_turn:
            user_message += self.game_basics.format(**state)
        if state.get("has_proposed"):
            user_message += self.proposal_prompt.format(**state)
        else:
            user_message += f"The other player said: '{state['last_message']}'\n" if state['last_message'] else "You are the first to play, there is no messages yet.\n"
        if self.chain_of_thought is not None:
            user_message += self.chain_of_thought.format(**state)
        return user_message
    
    def validate(self, response):
        """
        Validates the response from the LLM player.

        Args:
            response (str): The response from the LLM player.

        Returns:
            tuple: A tuple containing a boolean indicating validity and an error message if invalid.
        """
        errors = []
        
        # Check if reasoning tag exists
        if "<reason>" not in response or "</reason>" not in response:
            errors.append("Missing <reason>...</reason> tag.")
        
        # Check if message or propose tag exists, but not both
        has_message = "<message>" in response and "</message>" in response
        has_propose = "<propose>" in response and "</propose>" in response
        
        if has_message and has_propose:
            errors.append("Response contains both <message>...</message> and <propose>...</propose> tags. Only one is allowed per response.")
        elif not has_message and not has_propose:
            errors.append("Response must contain either <message>...</message> or <propose>...</propose> tag. Do not forget the closing tag.")
        
        # Check if propose tag is JSON parsable and follows the specified format
        if has_propose:
            propose_content = response.split("<propose>")[1].split("</propose>")[0].strip()
            try:
                propose_json = json.loads(propose_content)
                if not all(key in propose_json for key in ["i_take", "other_player_gets"]):
                    errors.append('The <propose> tag must contain JSON with keys "i_take" and "other_player_gets".')
                else:
                    i_take = propose_json["i_take"]
                    other_player_gets = propose_json["other_player_gets"]
                    
                    if not (isinstance(i_take, list) and len(i_take) == 3 and all(isinstance(x, int) for x in i_take)):
                        errors.append('The "i_take" value must be a list of 3 integers.')
                    
                    if not (isinstance(other_player_gets, list) and len(other_player_gets) == 3 and all(isinstance(x, int) for x in other_player_gets)):
                        errors.append('The "other_player_gets" value must be a list of 3 integers.')
            except json.JSONDecodeError:
                errors.append("The content within <propose>...</propose> is not valid JSON.")
        
        # Generate error message or return success
        if errors:
            return False, "Errors: " + "; ".join(errors)
        else:
            return True, "Response is valid."

    def extract(self, message):
        """
        Extracts the content from the response message.

        Args:
            message (str): The response message.

        Returns:
            tuple: A tuple containing a boolean indicating if it's a proposal and the extracted content.
        """
        pattern = r'<message>(.*?)</message>|<propose>(.*?)</propose>'
        match = re.search(pattern, message, re.DOTALL)

        if match.group(2):
            # Extract json from proposal
            return True, json.loads(match.group(2))["i_take"]
        else:
            return False, match.group(1)
    
    def reset_history(self):
        """
        Resets the message history of the LLM player.

        Returns:
            list: The message history before resetting.
        """
        self.first_turn = True
        history = self.dond_player.history
        self.dond_player.reset_messages()
        return history
