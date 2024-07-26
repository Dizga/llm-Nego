import gymnasium
import regex as re
import json

class DoND():
    def __init__(self,
            instructions: str):
        # Generate values for each player
        self.instructions = instructions
        self.turn = 0
        self.quantities = {'books': 5, 'hats': 4, 'balls': 3}
        self.a_values = {'books': 5, 'hats': 4, 'balls':3}
        self.b_values = {'books': 5, 'hats': 4, 'balls':3}
        self.has_proposed = False
        self.a_prop = {}
        self.b_prop = {}
        self.a_perspective = "You: "
        self.b_perspective = ""
        self.points_a = 0
        self.points_b = 0

    def step(self, output: str):
        "Play a move. Returns True if game ongoing, False when over."
        self.turn += 1

        # player has proposal or needs to propose
        if self.has_proposed:
            if self.propose(output) != False and self.verify_props_match():
                self.set_points()
            return False # game ended
        if self.propose(output):
            self.has_proposed = True
            if self.current_turn() == "a":
                self.b_perspective += output + "<You> : "
            return True

        # no proposal, continue conversation
        if self.current_turn() == "a":
            self.b_perspective += output + "<You> : "
            self.a_perspective += output + "<Other> : "
            return self.wrap()

        self.a_perspective += output + "<You> : "
        self.b_perspective += output + "<Other> : "
        return self.wrap()

    def verify_props_match(self):
        if self.a_prop['books']*self.b_prop['books']!=self.quantities['books']: return False
        if self.a_prop['hats']*self.b_prop['hats']!=self.quantities['hats']: return False
        if self.a_prop['balls']*self.b_prop['balls']!=self.quantities['balls']: return False

    def set_points(self):
        self.points_a = (self.a_values['books']*self.a_prop['books'] +
        self.a_values['hats']*self.a_prop['hats'] +
        self.a_values['balls']*self.a_prop['balls'])
        self.points_b =  (self.b_values['books']*self.b_prop['books'] +
        self.b_values['hats']*self.b_prop['hats'] +
        self.b_values['balls']*self.b_prop['balls'])

    def wrap(self) -> str:
        "Add instructions to model input string"
        if self.current_turn() == "a":
            return self.instructions + f"""<Game Start>
                    There is a total of {self.quantities['books']} books,
                    {self.quantities['hats']} hats and {self.quantities['balls']} balls.
                    You values are {self.a_values['book']} for a book,
                        {self.a_values['hat']} for a hat and {self.a_values['ball']} for a ball.
                """ + self.a_perspective

        return self.instructions + f"""<Game Start>
                There is a total of {self.quantities['books']} books,
                {self.quantities['hats']} hats and {self.quantities['balls']} balls.
                You values are {self.b_values['book']} for a book,
                    {self.b_values['hat']} for a hat and {self.b_values['ball']} for a ball.
            """ + self.b_perspective

    def check_if_message(self, string: str) -> bool:
        return re.search(r"\[ Message \] .*", string)

    def propose(self, string: str) -> bool:
        "Sets proposal."
        if re.search(r"\[ Proposal \] \{'books': \d+, 'hats': \d+, 'balls': \d+\}", string) == False:
            return False
        prop = json.loads(string)
        if prop['books'] > self.quantities['books']: return False
        if prop['hats'] > self.quantities['hats']: return False
        if prop['balls'] > self.quantities['balls']: return False
        if self.current_turn() == "a": self.a_prop = prop
        else: self.b_prop = prop
        return True

    def reset(self):
        self.turn = 0
        self.has_proposed = False
        self.a_perspective = "You: "
        self.b_perspective = ""

    def render(self):
        "Render the interations without chain of thought."

    def export(self):
        "Export the game in Json Format."

    def close(self):
        pass

    def current_turn(self):
        if self.turn % 2==0: return "a"
        return "b"
