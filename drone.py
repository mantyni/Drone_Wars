import pygame
import time

class Drone:
    def __init__(self, gameDisplay, display_width=800, display_height=600, *args, **kwargs):
        self.drone_speed = 20 # Default rate of change for drone movement
        self.x_change = 0
        self.y_change = 0
        self.x = 0
        self.y = 0
        self.drone_width = 70
        self.drone_height = 70
        self.display_width = display_width
        self.display_height = display_height
        self.gameDisplay = gameDisplay
        self.droneImg = pygame.image.load('images/drone.png')
        self.droneImg = pygame.transform.scale(self.droneImg, (int(self.display_width*0.08),int(self.display_height*0.1)))

    def move_left(self):
        self.x_change = -self.drone_speed

    def move_right(self):
        #self.x_change += self.drone_speed
        self.x_change = self.drone_speed

    def move_up(self):
        self.y_change = -self.drone_speed

    def move_down(self):
        self.y_change = +self.drone_speed

    def move_left_big(self):
        self.x_change -= 2* self.drone_speed

    def update(self):
        self.x += self.x_change
        self.y += self.y_change

    def draw(self):
        self.gameDisplay.blit(self.droneImg, (self.x,self.y))
  