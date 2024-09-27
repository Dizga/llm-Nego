import regex as re
import json
import random
import os


class DondGame:
    def __init__(
        self,
        mode="coop",
        max_turns=None,
        player_order="deterministic",
        setup="random_read",
        setups_file=None,
        rounds_per_game=10,
        items=None,
        player_0_values=None,
        player_1_values=None,
        quantities=None,
        finalization_visibility=False,
    ):
        """
        Initializes the DoND game.

        Args:
            mode (str): The mode of the game, either 'coop' or 'basic'.
            max_turns (int): The maximum number of turns allowed in the game.
            player_order (str): The order of players, either 'deterministic' or 'stochastic'.
            setup (str): The setup type, either 'random_read' or 'manual'.
            setups_file (str): The file containing game setups.
            rounds_per_game (int): The number of rounds per game.
            items (list): The list of items in the game.
            player_0_values (list): The values for player 0.
            player_1_values (list): The values for player 1.
            quantities (list): The quantities of items.
            finalization_visibility (bool): Visibility of finalization.
        """
        self.mode = mode
        self.max_turns = max_turns
        self.player_order = player_order
        self.setup = setup
        self.setups_file = setups_file
        self.rounds_per_game = rounds_per_game
        self.quantities = quantities
        self.finalization_visibility = finalization_visibility

        if self.setup == "random_read":
            self.items = ["books", "hats", "balls"]
            self.settings = []
            # Get dataset of game setups from file
            with open(self.setups_file) as f:
                lines = f.readlines()
                self.nb_settings = len(lines)
                for i in range(0, self.nb_settings, 2):
                    # TODO: ensure that quantities match!
                    l = [int(item) for item in lines[i].split()]
                    l2 = [int(item) for item in lines[i + 1].split()]
                    quantities = {
                        key: value for key, value in zip(self.items, [l[0], l[2], l[4]])
                    }
                    player_0_values = {
                        key: value for key, value in zip(self.items, [l[1], l[3], l[5]])
                    }
                    player_1_values = {
                        key: value for key, value in zip(self.items, [l2[1], l2[3], l2[5]])
                    }
                    self.settings.append((quantities, player_0_values, player_1_values))
            self.nb_settings = len(self.settings)

        elif self.setup == "manual":
            self.items = items
            self.quantities = {key: value for key, value in zip(self.items, quantities)}
            self.values_player_0 = {
                key: value for key, value in zip(self.items, player_0_values)
            }
            self.values_player_1 = {
                key: value for key, value in zip(self.items, player_1_values)
            }

        self.reset()

    def step(self, output, is_finalization=False) -> bool:
        """
        Advances the game by one step.

        Args:
            output (str | list): The output message or finalization list from the player.
            is_finalization (bool): Indicates if the output is a finalization.

        Returns:
            bool: False if game ended else True.
        """
        self.turn += 1

        self.last_message = output

        if self.has_finalized:  # Other player made finalization last turn

            if not is_finalization:  # player has not made a finalization after other player, automatic loss
                self.points_player_0 = 0
                self.points_player_1 = 0
                self.agreement_reached = False

            else:  # Player has made a finalization

                self.finalize(output)

                if self.verify_finalizations_match():  # If finalizations are complementary
                    self.set_points()
                    self.agreement_reached = True
                else:
                    self.points_player_0 = 0
                    self.points_player_1 = 0
                    self.agreement_reached = False

            self.end_round()
            return self.get_state()

        self.has_finalized = is_finalization

        if is_finalization:
            self.has_finalized = True
            self.finalize(output)

        if self.turn > self.max_turns:
            self.end_round()
            return self.get_state()  # round ended due to exceeding max turns

        return self.get_state()  # round not ended

    def verify_finalizations_match(self):
        """
        Verifies if the finalizations from both players match the total quantities.

        Returns:
            bool: True if the finalizations match, False otherwise.
        """
        for item in self.items:
            if self.player_0_prop[item] + self.player_1_prop[item] != self.quantities[item]:
                return False
        return True

    def set_points(self):
        """
        Sets the points for both players based on their finalizations.
        """
        points_player_0 = sum(
            self.values_player_0[item] * self.player_0_prop[item] for item in self.items
        )
        points_player_1 = sum(
            self.values_player_1[item] * self.player_1_prop[item] for item in self.items
        )

        if self.mode == "coop":
            total = points_player_0 + points_player_1
            self.points_player_0 = total
            self.points_player_1 = total

        elif self.mode == "basic":
            self.points_player_0 = points_player_0
            self.points_player_1 = points_player_1

    def finalize(self, finalization: list):
        """
        Records the finalization from the current player.

        Args:
            finalization (list): The list of finalized quantities for each item.
        """
        if self.current_turn() == "player_0":
            self.player_0_prop = finalization["i_take"]
        else:
            self.player_1_prop = finalization["i_take"]

    def get_state(self):
        """
        Retrieves the current state of the game.

        Returns:
            dict: The current state of the game.
        """
        out = {
            "mode": self.mode,
            "game_ended": self.game_ended,
            "round_ended": self.round_ended,
            "is_new_round": True if self.turn <= 2 else False,
            "is_new_game": True if (self.turn <= 2 and self.round_nb == 1) else False,
            "items": self.items,
            "turn": self.turn,
            "current_turn": self.current_turn(),
            "round_number": self.round_nb,
            "nb_rounds": self.rounds_per_game,
            "quantities": self.quantities,
            "has_finalized": self.has_finalized,
            "last_message": self.last_message,
            "finalization_visibility": self.finalization_visibility,
            "round_agreements": self.round_agreement_reached,
            "round_points": [self.round_points_player_0, self.round_points_player_1],
        }
        return out

    def get_play_order(self):
        """
        Get the order of players.

        Returns:
            list: The order of player indices.
        """
        if self.player_order == "deterministic":
            # If the order is deterministic, the players will always be [0, 1]
            return [0, 1]
        elif self.player_order == "stochastic":
            # If the order is stochastic, randomly shuffle the player order
            return random.sample([0, 1], 2)  # Randomly shuffle between [0, 1]
        else:
            # Handle invalid player_order values
            raise ValueError(
                f"Invalid player_order: {self.player_order}. Must be 'deterministic' or 'stochastic'."
            )

    def set_new_game_settings(self):
        """
        Sets new game settings based on the setup type.
        """
        if self.setup == "manual":
            return

        # Pick random trio of quantities & values from dataset
        elif self.setup == "random_read":
            setting_id = random.randint(0, self.nb_settings - 1)
            self.quantities, self.values_player_0, self.values_player_1 = self.settings[
                setting_id
            ]

    def archive_player_states(self):
        """
        Archives the states of the players for the current round.
        """
        self.round_player_0_prop.append(self.player_0_prop)
        self.round_player_1_prop.append(self.player_1_prop)
        self.round_points_player_0.append(self.points_player_0)
        self.round_points_player_1.append(self.points_player_1)
        self.round_values_player_0.append(self.values_player_0)
        self.round_values_player_1.append(self.values_player_1)
        self.round_quantities.append(self.quantities)
        self.round_agreement_reached.append(self.agreement_reached)

    def reset(self):
        """
        Resets the game to its initial state.

        Returns:
            tuple: The quantities of items and the values for player 0 and player 1.
        """
        self.has_finalized = False
        self.player_0_prop = {}
        self.player_1_prop = {}
        self.points_player_0 = 0
        self.points_player_1 = 0
        self.agreement_reached = False
        self.last_message = None
        self.round_nb = 1
        self.turn = 1
        self.round_ended = False
        self.game_ended = False
        self.last_message = None
        self.round_player_0_prop = []
        self.round_player_1_prop = []
        self.round_points_player_0 = []
        self.round_points_player_1 = []
        self.round_values_player_0 = []
        self.round_values_player_1 = []
        self.round_quantities = []
        self.round_agreement_reached = []
        self.set_new_game_settings()

    def end_round(self):
        """
        Ends the current round and prepares for the next round.
        """
        self.archive_player_states()
        self.round_nb += 1
        self.has_finalized = False
        self.player_0_prop = {}
        self.player_1_prop = {}
        self.points_player_0 = 0
        self.points_player_1 = 0
        self.agreement_reached = False
        self.last_message = None
        self.turn = 1
        self.round_ended = True
        self.game_ended = False
        self.last_message = None
        if self.round_nb > self.rounds_per_game:
            self.game_ended = True
        self.set_new_game_settings()

    def current_turn(self):
        """
        Determines the current player's turn.

        Returns:
            str: 'player_0' if it's player 0's turn, 'player_1' if it's player 1's turn.
        """
        return "player_0" if self.turn % 2 == 0 else "player_1"

    def export_game(self, file_path):
        """
        Exports the round history lists to a JSON file at the specified file path.

        Args:
            file_path (str): The path to the file where the JSON data will be saved.
        """
        folder_path = os.path.dirname(file_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        round_history = {
            "round_player_0_prop": self.round_player_0_prop,
            "round_player_1_prop": self.round_player_1_prop,
            "round_points_player_0": self.round_points_player_0,
            "round_points_player_1": self.round_points_player_1,
            "round_values_player_0": self.round_values_player_0,
            "round_values_player_1": self.round_values_player_1,
            "round_quantities": self.round_quantities,
            "round_agreement_reached": self.round_agreement_reached,
        }

        with open(file_path, "w") as f:
            json.dump(round_history, f, indent=4)