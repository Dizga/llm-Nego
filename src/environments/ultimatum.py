import json
import re


class Ultimatum:
    def __init__(self, max_turns=10, rounds_per_game=10, **kwargs):
        """
        Initializes the Ultimatum Game.

        Args:
            max_turns (int): The maximum number of turns allowed in a single round.
            rounds_per_game (int): The number of rounds in the game.
        """
        self.max_turns = max_turns
        self.rounds_per_game = rounds_per_game
        self.reset()

    def reset(self):
        """
        Resets the game to its initial state.

        Returns:
            dict: The initial state of the game.
        """
        self.turn = 1
        self.current_message = ""
        self.current_offer = None
        self.agreement_reached = False
        self.round_nb = 1
        self.game_ended = False
        self.points_player_0_history = []
        self.points_player_1_history = []
        self.agreement_reached_history = []
        # self.round_history = []

        return self.get_state()

    def step(self, output):
        """
        Advances the game by one step based on the player's output.

        Args:
            player (str): The player making the move ('player_0' or 'player_1').
            output (str): The output message or proposal from the player.

        Returns:
            dict: The updated state of the game.
        """
        message, proposition = self.extract_response(output)

        if self.is_aggreement_reached(proposition):
            self.agreement_reached = True
            self.end_round()
        else:
            self.current_message = message
            self.current_offer = proposition

        # self.log_move(player, response)
        self.turn += 1

        if self.turn > self.max_turns or self.agreement_reached:
            self.end_round()

        return self.get_state()

    def log_move(self, player, response):
        """
        Logs the player's move.

        Args:
            player (str): The player making the move ('player_0' or 'player_1').
            response (dict): The player's move details.
        """
        move_data = {
            "round": self.round_nb,
            "turn": self.turn,
            "player": player,
            "reason": response.get("reason", ""),
            "message": response.get("message", ""),
            "proposal": response.get("propose", "")
        }
        if player == "player_0":
            self.player_0_history.append(move_data)
        else:
            self.player_1_history.append(move_data)

    def is_aggreement_reached(self, proposal):
        """
        Checks if the proposal is accepted by both players.

        Args:
            proposal (str): The proposal to be checked.

        Returns:
            bool: True if the proposal is accepted, False otherwise.
        """

        if self.current_offer is None:
            return

        self.agreement_reached = self.current_offer['i_take'] == proposal['other_player_gets'] and self.current_offer['other_player_gets'] == proposal['i_take']

        if self.agreement_reached:
            if self.current_turn() == "player_0":
                self.points_player_0_history.append(self.current_offer['i_take'])
                self.points_player_1_history.append(self.current_offer['other_player_gets'])
            else:
                self.points_player_0_history.append(self.current_offer['other_player_gets'])
                self.points_player_1_history.append(self.current_offer['i_take'])


    def end_round(self):
        """
        Ends the current round, and prepares the game for the next round.
        """
        # self.round_history.append({
        #     "round_nb": self.round_nb,
        #     "player_0_moves": self.player_0_history,
        #     "player_1_moves": self.player_1_history,
        #     "agreement_reached": self.agreement_reached,
        #     "final_offer": self.current_offer if self.agreement_reached else None
        # })

        if not self.agreement_reached:
            self.points_player_0_history.append(0)
            self.points_player_1_history.append(0)
        self.agreement_reached_history.append(self.agreement_reached)

        self.round_nb += 1
        self.turn = 0
        self.agreement_reached = False
        self.current_message = None
        self.current_offer = None
        # self.player_0_history = []
        # self.player_1_history = []

        if self.round_nb > self.rounds_per_game:
            self.game_ended = True

    def extract_response(self, output):
        """
        Extracts the reason, message, and proposal from the player's output.

        Args:
            output (str): The player's output.

        Returns:
            dict: Extracted reason, message, and proposal.
        """
        message_match = re.search(r'<message>(.*?)</message>', output, re.DOTALL)
        propose_match = re.search(r'<propose>(.*?)</propose>', output, re.DOTALL)

        message = message_match.group(1).strip() if message_match else ""

        propose_str = propose_match.group(1).strip() if propose_match else ""
        
        # Try to parse the proposal as JSON
        try:
            propose = json.loads(propose_str)
        except json.JSONDecodeError:
            propose = propose_str  # If parsing fails, treat it as a string

        return message, propose

    def get_state(self):
        """
        Retrieves the current state of the game.

        Returns:
            dict: The current state of the game.
        """
        return {
            "game_ended": self.game_ended,
            "round_number": self.round_nb,
            "turn": self.turn,
            "max_turns": self.max_turns,
            "last_message": self.current_message,
            "last_proposition": self.current_offer,
            "agreement_reached": self.agreement_reached
        }
    
    # def export(self):
    #     """
    #     Export game and round metrics.
    #     """
    #     rounds = []
    #     for round_id in range(self.rounds_per_game):
    #         rounds.append(self.export_round(round_id))
    #     summary = self.export_summary()
    #     return summary, rounds
    
    def export(self):
        """
        Export round metrics.
        """
        rounds = []
        for round_id in range(self.rounds_per_game):
            rounds.append(self.export_round(round_id))
        return rounds

    def export_summary(self):
        return {}

    def export_round(self, id=-1):
        """
        Exports the current state of the game as a dictionary.

        Returns:
            dict: The current state of the game.
        """
        return {
            'round_id': id,
            'player_0_score': self.points_player_0_history[id],
            'player_1_score': self.points_player_1_history[id],
            'player_0_return': sum(self.points_player_0_history[id:]),
            'player_1_return': sum(self.points_player_1_history[id:]),
            'agreement_reached': self.agreement_reached_history[id],
        }
    
    def current_turn(self):
        """
        Determines the current player's turn.

        Returns:
            str: 'player_0' if it's player 0's turn, 'player_1' if it's player 1's turn.
        """
        return "player_0" if self.turn % 2 == 0 else "player_1"