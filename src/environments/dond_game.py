import regex as re
import json
import random

class DondGame:
    def __init__(self, 
                 max_turns=None,
                 setup="random_read",
                 setups_file=None,
                 nb_rounds = 10
                 ):
        """
        Initializes the DoND game.

        Args:
            max_turns (int): The maximum number of turns allowed in the game.
        """
        self.max_turns = max_turns
        self.setup = setup
        self.setups_file = setups_file
        self.nb_rounds = nb_rounds

        if self.setups_file is not None:
            self.settings = []
            # Get dataset of game setups from file
            with open(self.setups_file) as f:
                lines = f.readlines()
                self.nb_settings = len(lines)
                for i in range(0, self.nb_settings, 2):
                    # TODO: ensure that quantities match!
                    l = [int(item) for item in lines[i].split()]
                    l2 = [int(item) for item in lines[i+1].split()]
                    quantities = {key: value for key, value in zip(self.items, [l[0], l[2], l[4]])}
                    p0_values = {key: value for key, value in zip(self.items, [l[1], l[3], l[5]])}
                    p1_values = {key: value for key, value in zip(self.items, [l2[1], l2[3], l2[5]])}
                    self.settings.append((quantities, p0_values, p1_values))

        self.reset()


    def reset(self):
        """
        Resets the game to its initial state.

        Returns:
            tuple: The quantities of items and the values for player 0 and player 1.
        """
        self.turn = 0
        self.items = ['books', 'hats', 'balls']
        self.quantities = None
        self.has_proposed = False

        # Get hard coded quantities & values
        if self.setup == "primitive":
            self.quantities = {key: value for key, value in zip(self.items, [5, 4, 3])}
            self.values_p0 = {key: value for key, value in zip(self.items, [5, 4, 3])}
            self.values_p1 = {key: value for key, value in zip(self.items, [3, 4, 5])}

        # Pick random trio of quantities & values from dataset
        else:
            setting_id = random.randint(0, self.nb_settings-1)
            self.quantities, self.values_p0, self.values_p1 = self.settings[setting_id]

        self.reset_player_states()
        self.round_nb = 1
        self.new_round = True
        self.game_ended = False

        return self.quantities, self.values_p0, self.values_p1
    
    def reset_player_states(self):
        self.p0_prop = {}
        self.p1_prop = {}
        self.points_p0 = 0
        self.points_p1 = 0
        self.agreement_reached = False
        self.last_message = ""

    def archive_player_states(self):
        self.p0_prop_history.append(self.p0_prop)
        self.p1_prop_history.append(self.p1_prop)
        self.points_p0_history.append(self.points_p0)
        self.points_p1_history.append(self.points_p1)
        self.agreement_reached_history.append(self.agreement_reached)

    def end_round(self):
        self.round_nb += 1
        self.archive_player_states()
        self.reset_player_states()
        self.new_round = True
        if self.rounds_player > self.nb_rounds:
            self.game_ended = True

    
    def step(self, output, is_proposal=False)-> bool: 
        """
        Advances the game by one step.

        Args:
            output (str | list): The output message or proposal list from the player.
            is_proposal (bool): Indicates if the output is a proposal.

        Returns:
            bool: False if game ended else True.
        """
        
        self.last_message = output

        if self.has_proposed:
            if not is_proposal: # player has not made a proposal after other player, automatic loss
                self.end_round()
                return self.game_ended
            self.propose(output)
            if self.verify_props_match():
                self.set_points()
                self.agreement_reached = True
                self.end_round()
            return self.game_ended  

        self.has_proposed = is_proposal

        if is_proposal:
            self.has_proposed = True
            self.propose(output)
        
        self.turn += 1

        if self.turn > self.max_turns:
            self.end_round()
            return self.game_ended  # round ended due to exceeding max turns

        return self.game_ended  # game not ended

    def get_state(self, player="current_turn"):
        """
        Retrieves the current state of the game.

        Args:
            player (str): The player whose state is to be retrieved ('p0', 'p1', or 'current_turn').

        Returns:
            dict: The current state of the game.
        """
        if player == "current_turn":
            player = self.current_turn()
        out = {
            'game_ended': self.game_ended,
            "new_round": self.new_round,
            "round_number": self.round_number,
            "book_cnt": self.quantities["books"],
            "hat_cnt": self.quantities["hats"],
            "ball_cnt": self.quantities["balls"],
            "agreement_reached": self.agreement_reached,
            "has_proposed": self.has_proposed,
            "last_message": self.last_message
        }
        if player == "p0":
            out["book_val"] = self.values_p0["books"]
            out["hat_val"] = self.values_p0["hats"]
            out["ball_val"] = self.values_p0["balls"]
            return out

        out["book_val"] = self.values_p1["books"]
        out["hat_val"] = self.values_p1["hats"]
        out["ball_val"] = self.values_p1["balls"]
        return out

    def verify_props_match(self):
        """
        Verifies if the proposals from both players match the total quantities.

        Returns:
            bool: True if the proposals match, False otherwise.
        """
        for item in self.items:
            if self.p0_prop[item] + self.p1_prop[item] != self.quantities[item]:
                return False
        return True

    def set_points(self):
        """
        Sets the points for both players based on their proposals.
        """
        self.points_p0 = sum(self.values_p0[item] * self.p0_prop[item] for item in self.items)
        self.points_p1 = sum(self.values_p1[item] * self.p1_prop[item] for item in self.items)

    def propose(self, proposal: list):
        """
        Records the proposal from the current player.

        Args:
            proposal (list): The list of proposed quantities for each item.
        """
        if self.current_turn() == "p0":
            self.p0_prop = proposal
        else:
            self.p1_prop = proposal

    def render(self):
        """
        Renders the current state of the game (not implemented).
        """
        pass

    def export(self):
        """
        Export game and round metrics.
        """
        rounds = []
        for round_id in range(self.nb_rounds):
            rounds.append(self.export_round(round_id))
        summary = self.export_summary()
        return summary, rounds

    def export_summary(self):
        return {
            'p0_total_reward': sum(self.points_p0_history),
            'p1_total_reward': sum(self.points_p1_history),
        }

    def export_round(self, id=-1):
        """
        Exports the current state of the game as a dictionary.

        Returns:
            dict: The current state of the game.
        """
        return {
            'round_id': id,
            'p0_score': self.points_p0_history[id],
            'p1_score': self.points_p1_history[id],
            'p0_return': sum(self.points_p0_history[id:]),
            'p1_return': sum(self.points_p1_history[id:]),
            'quantities': self.quantities_history[id],
            'p0_values': self.values_p0_history[id],
            'p1_values': self.values_p1_history[id],
            'p0_proposal': self.p0_prop_history[id],
            'p1_proposal': self.p1_prop_history[id],
            'agreement_reached': self.agreement_reached_history[id],
        }

    def current_turn(self):
        """
        Determines the current player's turn.

        Returns:
            str: 'p0' if it's player 0's turn, 'p1' if it's player 1's turn.
        """
        return "p0" if self.turn % 2 == 0 else "p1"
