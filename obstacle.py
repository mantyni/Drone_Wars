import pygame
import random

class Obstacle:
    def __init__(self, gameDisplay, display_width=800, display_height=600, *args, **kwargs):
        self.x = random.randrange(0, display_width)
        self.y = -100 # Give more space for obstacle to fully render
        self.speed = 40
        self.height = 100 # self.display_width / 8
        self.width = 100 # self.display_width / 6
        self.display_width = display_width
        self.display_height = display_height
        self.gameDisplay = gameDisplay
        self.img = pygame.image.load('images/asteroid.png') #.convert() # comment out convert for nice png
        self.img = pygame.transform.scale(self.img, (int(self.display_width*0.16),int(self.display_height*0.2)))
        #self.img = pygame.transform.scale(self.img, (int(self.display_width*0.08),int(self.display_height*0.1)))


    def reset(self):
        self.x = random.randrange(0, self.display_width)
        self.y = 0 - self.height


    def update(self):
        self.y += self.speed


    def draw(self):
        self.gameDisplay.blit(self.img, (self.x,self.y))