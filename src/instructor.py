import regex as re

class Instructor:
    "The instructor acts as a middle-man between the game and a LLM player."
    def __init__(self):
        pass

class DoNDInstructor(Instructor):
    def __init__(self, game_intro_file, chain_of_thought_file, dond_game, dond_player, player_type="p0"):
        self.first_turn = True
        
        with open(game_intro_file, 'r') as file:
            self.game_basics = file.read()
        
        self.chain_of_thought = None
        if chain_of_thought_file:
            with open(chain_of_thought_file, 'r') as file:
                self.chain_of_thought = file.read()
        
        self.dond_game = dond_game
        self.dond_player = dond_player
        self.player_type = player_type

    def play_move(self):
        state = self.dond_game.get_state()
        if state is None:
            return False
        
        user_message = self.get_usr_message(state)
        response = self.dond_player.prompt(user_message)
        
        # while not response:
        #     response = self.verificator(self.dond_player.prompt(user_message))
        
        is_proposal, content = self.extract(response)

        ongoing = self.dond_game.step(content, is_proposal) # whether the game is finished or not
        self.first_turn = False
        return ongoing

    def verificator(self, message):
        # TODO: add conditions that return false if message not correct
        return message

    def get_usr_message(self, state):
        # Get message from instructor
        user_message = ""
        if self.first_turn:
            user_message += self.game_basics 
            user_message += self.get_stringed_metrics(state["quantities"], state["values"])
        if state.get("has_proposed"): 
            user_message += "THE OTHER PLAYER HAS MADE A PROPOSAL."
        else:
            user_message += f"Other Player Reply: '{state['last_message']}'\n" if state['last_message'] else ""
        if self.chain_of_thought is not None:
            user_message += self.chain_of_thought
        return user_message

    def get_stringed_metrics(self, quantities, values):
        return (
            f"There is a total of {quantities['books']} books, "
            f"{quantities['hats']} hats, and {quantities['balls']} balls. "
            f"Your values are {values['books']} for a book, "
            f"{values['hats']} for a hat, and {values['balls']} for a ball."
        )

    def extract_DoND_message(self, response):
        pattern = r'<message>(.*?)</message>'
        match = re.search(pattern, response, re.DOTALL)
        return match.group(1) if match else None

    def check_DoND_conformity(self, message):
        if self.chain_of_thought:
            regex = r"<reason>(.*?)</reason>\s*(<message>(.*?)</message>|<proposal>(.*?)</proposal>)"
        else:
            regex = r"(<message>(.*?)</message>|<proposal>(.*?)</proposal>)"
        return re.match(regex, message) is not None

    def extract(self, message):

        if self.chain_of_thought:
            regex = r"<reason>(.*?)</reason>\s*(.*?)\s*(<message>(.*?)</message>|<proposal>(.*?)</proposal>)"
        else:
            regex = r"(<message>(.*?)</message>|<proposal>(.*?)</proposal>)"

        if re.match(regex, message) is None:
            return None, ''

        pattern = r'<message>(.*?)</message>|<proposal>(.*?)</proposal>'
        match = re.search(pattern, message, re.DOTALL)

        return bool(match.group(2)), match.group(2) if match.group(2) else match.group(1)
