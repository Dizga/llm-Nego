import regex as re
import json

class DoND:
    def __init__(self):
        self.reset()

    def reset(self):
        self.turn = 0
        self.quantities = {'books': 5, 'hats': 4, 'balls': 3}
        self.values_p0 = {'books': 5, 'hats': 4, 'balls': 3}
        self.values_p1 = {'books': 5, 'hats': 4, 'balls': 3}
        self.has_proposed = False
        self.p0_prop = {}
        self.p1_prop = {}
        self.points_p0 = 0
        self.points_p1 = 0
        self.agreement_reached = False
        self.last_message = ""
        return self.quantities, self.values_p0, self.values_p1

    def step(self, output: str):
        self.turn += 1
        self.last_message = output
        if self.has_proposed:
            if self.propose(output) and self.verify_props_match():
                self.set_points()
                self.agreement_reached = True
                return False
            return False  # Game ended with failure to be complementary
        
        if self.propose(output):
            self.has_proposed = True
            return True  # Continue the game
        
        if re.match(r"\[ Message \] .*", output):
            return True  # Continue the game
        
        return False  # Game ended due to bad formatting
    
    def get_state(self, player="current_turn"):
        "Returns True if other player has proposed or no move played."
        "Returns False if game is ended."
        "Else returns his last message."
        if player=="current_turn": player = self.current_turn()
        out = {
            "quantities": self.agreement_reached,
            "has_proposed": self.has_proposed,
            "last_message": self.last_message
        }
        if player=="p0":
            out["values"] = self.values_p0
            return out
        out["values"] = self.values_p1

    def verify_props_match(self):
        for item in self.quantities:
            if self.p0_prop[item] * self.p1_prop[item] != self.quantities[item]:
                return False
        return True

    def set_points(self):
        self.points_p0 = sum(self.values_p0[item] * self.p0_prop[item] for item in self.quantities)
        self.points_p1 = sum(self.values_p1[item] * self.p1_prop[item] for item in self.quantities)

    def propose(self, string: str) -> bool:
        "Determines if there is a valid proposal in the string."
        match = re.match(r"\[ Proposal \] \{.*\}", string)
        if not match:
            return False
        prop = json.loads(match.group(0)[12:])
        if any(prop[item] > self.quantities[item] for item in self.quantities):
            return False
        if self.current_turn() == "p0":
            self.p0_prop = prop
        else:
            self.p1_prop = prop
        return True

    def render(self):
        pass

    def export_game(self):
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
        return "p0" if self.turn % 2 == 1 else "p1"