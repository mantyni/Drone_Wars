import pygame

class Drone:
    
    def __init__(self, gameDisplay, display_width=800, display_height=600, drone_id = 0, *args, **kwargs):
        self.id = drone_id
        self.drone_speed = 20 # Default rate of change for drone movement
        self.x_change = 0
        self.y_change = 0
        self.drone_width = 70 
        self.drone_height = 70 
        self.display_width = display_width
        self.display_height = display_height
        self.gameDisplay = gameDisplay
        self.img = pygame.image.load('images/drone1.png').convert() # To fix png files use: pngcrush -ow -rem allb -reduce file.png
        self.img = pygame.transform.scale(self.img, (int(self.display_width*0.1),int(self.display_height*0.12)))
        self.done = False
        self.reward = 0
        
        # Manually assigning drone position based on ID to hardcode the screen side. 
        # The policy learns to control the drone on the left or right side of the screen. 
        # If positionts of drones are randomized, network struggles to differentiate between drones. 
        if drone_id == 0: # assign drone positions on the screen
            self.x = display_width * 0.8
            self.y = display_height * 0.85
        if drone_id == 1:
            self.x = display_width * 0.2
            self.y = display_height * 0.7

    def move_left(self):
        self.x_change = -self.drone_speed

    def move_right(self):
        self.x_change = self.drone_speed

    def move_up(self):
        self.y_change = -self.drone_speed

    def move_down(self):
        self.y_change = +self.drone_speed

    def update(self):
        self.x += self.x_change
        self.y += self.y_change
        self.x_change = 0
        self.y_change = 0

    def draw(self):
        self.gameDisplay.blit(self.img, (self.x,self.y))
