class World:
    def __init__(self, agents):
        self.agents = agents
        self.time = 0

    def apply_action(self, agent, action):
        if action == "give":
            agent.resources -= 1
        elif action == "trade":
            agent.resources += 1
