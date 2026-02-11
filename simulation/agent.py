import random

class Agent:
    def __init__(self, agent_id):
        self.id = agent_id
        self.resources = random.randint(5, 20)
        self.alive = True

    def decide_action(self, world):
        if self.resources <= 0:
            self.alive = False
            return None

        return random.choice(["trade", "give", "do_nothing"])
