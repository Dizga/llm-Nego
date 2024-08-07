import regex as re
import json
import random

class DondGame:
    def __init__(self, 
                 max_turns=None,
                 setup="random_read",
                 setups_file=None):
        """
        Initializes the DoND game.

        Args:
            max_turns (int): The maximum number of turns allowed in the game.
        """
        self.max_turns = max_turns
        self.setup = setup
        self.setups_file = setups_file
        self.line_in_setups_file = 7000  # TODO: automate
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


        if self.setup == "primitive":
            self.quantities = {key: value for key, value in zip(self.items, [5, 4, 3])}
            self.values_p0 = {key: value for key, value in zip(self.items, [5, 4, 3])}
            self.values_p1 = {key: value for key, value in zip(self.items, [3, 4, 5])}

        if self.setup == "random_read":
            # get setup from random read
            random_pair_id = random.randint(0, self.line_in_setups_file // 2) * 2
            with open(self.setups_file) as f:
                lines = f.readlines()[random_pair_id:random_pair_id + 2]
                for i in range(2):
                    l = [int(item) for item in lines[i].split()]
                    quantities = {key: value for key, value in zip(self.items, [l[0], l[2], l[4]])}
                    values = {key: value for key, value in zip(self.items, [l[1], l[3], l[5]])}
                    if i == 0:
                        self.values_p0 = values
                    else:
                        self.values_p1 = values
                    if self.quantities and self.quantities != quantities:
                        raise RuntimeError("Bad pair of DonD values/quantities.")
                    self.quantities = quantities

        self.p0_prop = {}
        self.p1_prop = {}
        self.points_p0 = 0
        self.points_p1 = 0
        self.agreement_reached = False
        self.last_message = ""
        return self.quantities, self.values_p0, self.values_p1

    def step(self, output, is_proposal=False):
        """
        Advances the game by one step.

        Args:
            output (str | list): The output message or proposal list from the player.
            is_proposal (bool): Indicates if the output is a proposal.

        Returns:
            bool: Whether the game should continue or not.
        """
        
        self.last_message = output

        if self.has_proposed:
            if not is_proposal:
                return False  # player has not made a proposal after other player, automatic loss
            self.propose(output)

            if self.verify_props_match():
                self.set_points()
                self.agreement_reached = True
                return False  # Game ended successfully
            return False  # Game ended with failure to be complementary

        self.has_proposed = is_proposal

        if is_proposal:
            self.has_proposed = True
            self.propose(output)
        
        self.turn += 1

        if self.turn > self.max_turns:
            return False  # Game ended due to exceeding max turns

        return True  # Continue the game

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
        Exports the current state of the game as a dictionary.

        Returns:
            dict: The current state of the game.
        """
        return {
            'p0_score': self.points_p0,
            'p1_score': self.points_p1,
            'quantities': self.quantities,
            'p0_values': self.values_p0,
            'p1_values': self.values_p1,
            'p0_proposal': self.p0_prop,
            'p1_proposal': self.p1_prop,
            'agreement_reached': self.agreement_reached,
        }

    def current_turn(self):
        """
        Determines the current player's turn.

        Returns:
            str: 'p0' if it's player 0's turn, 'p1' if it's player 1's turn.
        """
        return "p0" if self.turn % 2 == 0 else "p1"
