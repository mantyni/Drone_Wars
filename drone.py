import pygame
import random

class Drone:
    def __init__(self, gameDisplay, display_width=800, display_height=600, drone_id = 0, *args, **kwargs):
        self.id = drone_id
        self.drone_speed = 20 # Default rate of change for drone movement
        self.x_change = 0
        self.y_change = 0
        #self.x = display_width * random.uniform(0.15,0.85) # test if works - need to see if networks learns which drone to control. If doesn't work then pasas an argument of which drone it is. 
        self.y = display_height * 0.85
        self.drone_width = 70 
        self.drone_height = 70 
        self.display_width = display_width
        self.display_height = display_height
        self.gameDisplay = gameDisplay
        self.img = pygame.image.load('images/drone1.png').convert() # To fix up png files use: pngcrush -ow -rem allb -reduce file.png
        self.img = pygame.transform.scale(self.img, (int(self.display_width*0.1),int(self.display_height*0.12)))
        #self.img = pygame.transform.scale(self.img, (int(self.display_width*0.05),int(self.display_height*0.06)))

        if drone_id == 0: # assign drone positions on the screen
            self.x = display_height * 0.8
        if drone_id == 1:
            self.x = display_height * 0.2


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
