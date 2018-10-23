from agent import Drone_agent

class Policy():
    def _init_(self):
        self.move_left()

    def move_always_left(self):
        self.move_left()
        print("WORKS")
