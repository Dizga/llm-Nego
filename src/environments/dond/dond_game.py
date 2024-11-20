import regex as re
import json
import random
import os
from collections import deque
import copy

class DondGame:
    def __init__(
        self,
        players,
        mode="coop",
        max_turns=None,
        rounds_per_game=1,
        random_setup_func=None,
        random_setup_kwargs=None,
        role_assignator_func=None,
        role_assignator_func_kwargs=None,
        finalization_visibility=False,
        other_values_visibility=False,
    ):
        """
        Initializes the DoND game.

        Args:
            players (list): List of player names.
            mode (str): The mode of the game, either 'coop' or 'basic'.
            max_turns (int): The maximum number of turns allowed in the game.
            player_order (str): The order of players, either 'deterministic' or 'stochastic'.
            setup (str): The setup type, either 'random_read' or 'manual'.
            setups_file (str): The file containing game setups.
            rounds_per_game (int): The number of rounds per game.
            items (list): The list of items in the game.
            quantities (list): The quantities of items.
            finalization_visibility (bool): Visibility of finalization.
            other_values_visibility (bool): Visibility of other player's values.
        """

        self.players = players
        self.roles = ["starting_negotiator", "responding_negotiator"]
        self.mode = mode
        self.max_turns = max_turns
        self.random_setup_func = globals()[random_setup_func]
        self.random_setup_kwargs = random_setup_kwargs
        self.finalization_visibility = finalization_visibility
        self.rounds_per_game = rounds_per_game
        self.role_assignator_func = globals()[role_assignator_func]
        self.role_assignator_func_kwargs = role_assignator_func_kwargs
        self.other_values_visibility = other_values_visibility

        self.reset()

    def set_new_setup(self):
        """
        # TODO: write config
        """
        self.items, self.quantities, role_values = self.random_setup_func(**self.random_setup_kwargs)
        self.role_values = {
            self.roles[0]: role_values[0],
            self.roles[1]: role_values[1]
        }


    def step(self, action):
        """
        Advances the game by one step.

        Args:
            action (tuple): A tuple containing is_finalization and output.

        Returns:
            tuple: (observation, reward, done, info)
        """
        is_finalization, output = action
        self.turn += 1
        self.last_message = output
        self.round_ended = False
        self.is_new_round = True if self.turn <= 2 else False
        self.is_new_game = True if (self.turn <= 1 and self.round_nb == 0) else False
        self.game_over = False
        round_over = False

        if self.has_finalized:
            if not is_finalization:
                self.points = {player: 0 for player in self.players}
                self.agreement_reached = False
            else:
                self.finalize(output)
                if self.verify_finalizations_match():
                    self.set_points()
                    self.agreement_reached = True
                else:
                    self.points = {player: 0 for player in self.players}
                    self.agreement_reached = False
            round_over = True

        else:
            if is_finalization:
                self.has_finalized = True
                self.finalize(output)

            if self.turn > self.max_turns:
                round_over = True

        self.role_deque.rotate(-1)
        if round_over: 
            self.new_round()
        if self.round_nb > self.rounds_per_game-1:
            self.game_over = True

        
        state = self.get_state()
        reward = None
        done = self.game_over
        info = self.get_info()  

        return state, reward, done, info

    def render(self, mode='human'):
        """
        Render the current state of the game.
        """
        print(f"Current state: {self.get_state()}")

    def close(self):
        """
        Clean up resources.
        """
        pass

    def verify_finalizations_match(self):
        """
        Verifies if the finalizations from both players match the total quantities.

        Returns:
            bool: True if the finalizations match, False otherwise.
        """
        for item in self.items:
            total = sum(self.role_props[role][item] for role in self.roles)
            if total != self.quantities[item]:
                return False
        return True

    def set_points(self):
        """
        Sets the points for each role based on their finalizations.
        """
        utilities = {
            role: sum(self.role_values[role][item] * self.role_props[role][item] for item in self.items)
            for role in self.roles
        }

        if self.mode == "coop":
            total = sum(utilities.values())
            self.points = {role: total for role in self.roles}

        elif self.mode == "basic":
            self.points = {role: utilities[role] for role in self.roles}

    def finalize(self, finalization: list):
        """
        Records the finalization from the current player.

        Args:
            finalization (list): The list of finalized quantities for each item.
        """
        current_role = self.current_turn()
        self.role_props[current_role] = finalization["i_take"]

    def get_state(self):
        """
        Retrieves the current state of the game.

        Returns:
            dict: The current state of the game.
        """
        state = {
            "mode": self.mode,
            "role_values": self.role_values,
            "role_props": self.role_props,
            "player_to_role": self.player_to_role,
            "is_new_round": self.is_new_round,
            "is_new_game": self.is_new_game,
            "game_over": self.game_over,
            "items": self.items,
            "turn": self.turn,
            "max_turns": self.max_turns,
            "current_player": self.get_current_player(),
            "round_number": self.round_nb,
            "nb_rounds": self.rounds_per_game,
            "quantities": self.quantities,
            "has_finalized": self.has_finalized,
            "last_message": self.last_message,
            "players" : self.players,
            "finalization_visibility": self.finalization_visibility,
            "other_values_visibility": self.other_values_visibility,
            # rounds history
            "round_player_roles": self.round_player_roles,
            "round_quantities": self.round_quantities,
            "round_values": self.round_values,
            "round_finalizations": self.round_finalizations,
            "round_agreements_reached": self.round_agreements_reached,
            "round_points": self.round_points,
        }
        return state
    
    def get_info(self):
        return {
            "mode": self.mode,
            "players" : self.players,
            "finalization_visibility": self.finalization_visibility,
            "other_values_visibility": self.other_values_visibility,
            "round_player_roles": self.round_player_roles,
            "round_quantities": self.round_quantities,
            "round_values": self.round_values,
            "round_finalizations": self.round_finalizations,
            "round_agreements_reached": self.round_agreements_reached,
            "round_points": self.round_points,
        }

    def archive_player_states(self):
        """
        Archives the states of the players for the current round.
        """
        # Ensure points are initialized for all roles
        if not all(role in self.points for role in self.roles):
            self.points = {role: 0 for role in self.roles}
        
        self.round_player_roles.append(self.player_to_role.copy())
        self.round_quantities.append(self.quantities)
        self.round_values.append({role: self.role_values[role] for role in self.roles})
        self.round_finalizations.append({role: self.role_props[role] for role in self.roles})
        self.round_agreements_reached.append(self.agreement_reached)
        self.round_points.append({role: self.points[role] for role in self.roles})

    def new_round(self):
        """
        Ends the current round and prepares for the next round.
        """
        self.archive_player_states()
        self.round_nb += 1
        self.has_finalized = False
        self.role_props = {role: {} for role in self.roles}
        self.points = {role: 0 for role in self.roles}  # Ensure points are reset
        self.agreement_reached = False
        self.last_message = None
        self.turn = 0
        self.last_message = None
        self.set_new_setup()
        self.assign_roles()
        self.role_deque = deque(self.roles)


    def reset(self, checkpoint=None):
        """
        Resets the game to its initial state or to a checkpoint if provided.

        Args:
            checkpoint (dict, optional): A dictionary containing the checkpoint state.
        """
        if checkpoint:
            self.load_checkpoint(checkpoint)
        else:
            self.has_finalized = False
            self.role_props = {role: {} for role in self.roles}
            self.points = {role: 0 for role in self.roles}  # Ensure points are initialized
            self.agreement_reached = False
            self.last_message = None
            self.round_nb = 0
            self.turn = 0
            self.is_new_round = True
            self.is_new_game = True
            self.game_over = False
            self.last_message = None
            self.role_deque = deque(self.roles)
            self.player_to_role = None
            self.round_player_roles = [] 
            self.round_quantities = []
            self.round_values = []        
            self.round_finalizations = [] 
            self.round_agreements_reached = [] 
            self.round_points = []
            self.set_new_setup()
            self.assign_roles()

    def get_current_player(self):
        """
        Get the current player (the one who has to play next)
        """
        if not hasattr(self, 'role_to_player') or not hasattr(self, 'role_deque') or not self.role_deque:
            return None
        return self.role_to_player[self.role_deque[0]]

    def current_turn(self):
        """
        Determines the current role's turn.

        Returns:
            str: The name of the current role.
        """
        return self.role_deque[0]

    def assign_roles(self):
        """
        Assigns roles to players for the current round using the role_assignator_func.
        """
        self.player_to_role = self.role_assignator_func(self.get_state(), **self.role_assignator_func_kwargs)
        
        # Create player_to_role mapping
        self.role_to_player = {role: player for player, role in self.player_to_role.items()}

    def load_checkpoint(self, checkpoint):
        """
        Loads the game state from a checkpoint.

        Args:
            checkpoint (dict): A dictionary containing the checkpoint state.
        """
        self.__dict__.update(checkpoint)

def uniform_quant_random_vals(items, min_quant, max_quant, min_val, max_val):
    quant = random.randint(min_quant, max_quant)
    val_starting_negotiator = [random.randint(min_val, max_val) for _ in range(quant)]
    val_responding_negotiator = copy.deepcopy(val_starting_negotiator)
    random.shuffle(val_responding_negotiator)
    val_starting_negotiator = {item: val for item, val in zip(items, val_starting_negotiator)}
    val_responding_negotiator = {item: val for item, val in zip(items, val_responding_negotiator)}
    quantities = {item:q for item,q in zip(items, [quant]*len(items))}
    return items, quantities, (val_starting_negotiator, val_responding_negotiator)

def independent_random_vals(items, min_quant, max_quant, min_val, max_val):
    quantities = {item: random.randint(min_quant, max_quant) for item in items}
    val_starting_negotiator = {item: random.randint(min_val, max_val) for item in items}
    val_responding_negotiator = {item: random.randint(min_val, max_val) for item in items}
    return items, quantities, (val_starting_negotiator, val_responding_negotiator)

def fixed_manual(items, quantities, val_starting_negotiator, val_responding_negotiator):
    quantities = {item: q for item, q in zip(items, quantities)}
    val_starting_negotiator = {item: v for item, v in zip(items, val_starting_negotiator)}
    val_responding_negotiator = {item: v for item, v in zip(items, val_responding_negotiator)}
    return items, quantities, (val_starting_negotiator, val_responding_negotiator)

def random_quant_fixed_vals(items, min_quant, max_quant, val_starting_negotiator, val_responding_negotiator):
    quantities = {item: random.randint(min_quant, max_quant) for item in items}
    val_starting_negotiator = {item: v for item, v in zip(items, val_starting_negotiator)}
    val_responding_negotiator = {item: v for item, v in zip(items, val_responding_negotiator)}
    return items, quantities, (val_starting_negotiator, val_responding_negotiator)

def alternating_role_assignator(state, **kwargs):
    """
    Alternates roles between player_0 and player_1 at each round.
    At the first round, player_0 is assigned to the role "starting_negotiator".

    Args:
        state (dict): The current state of the game.
        kwargs (dict): Additional keyword arguments (not used here).

    Returns:
        dict: A mapping of players to roles.
    """
    round_number = state["round_number"]
    players = state["players"]
    roles = ["starting_negotiator", "responding_negotiator"]

    if round_number % 2 == 0:
        # Even rounds: player_0 is "starting_negotiator"
        player_to_role = {players[0]: roles[0], players[1]: roles[1]}
    else:
        # Odd rounds: player_1 is "starting_negotiator"
        player_to_role = {players[0]: roles[1], players[1]: roles[0]}

    return player_to_role


def fixed_role_assignator(state, **kwargs):
    """
    Always assigns player_0 to the role "starting_negotiator".

    Args:
        state (dict): The current state of the game.
        kwargs (dict): Additional keyword arguments (not used here).

    Returns:
        dict: A mapping of players to roles.
    """
    players = state["players"]
    roles = ["starting_negotiator", "responding_negotiator"]

    # Always assign player_0 to "starting_negotiator"
    player_to_role = {players[0]: roles[0], players[1]: roles[1]}

    return player_to_role