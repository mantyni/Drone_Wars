import pygame
import random

class Obstacle:
    def __init__(self, gameDisplay, display_width=800, display_height=600, *args, **kwargs):
        self.x = 0
        self.y = 0
        self.speed = 30
        self.height = 100
        self.width = 100
        self.display_width = display_width
        self.display_height = display_height
        self.gameDisplay = gameDisplay
        self.img = pygame.image.load('images/asteroid.png')
        self.img = pygame.transform.scale(self.img, (int(self.display_width*0.16),int(self.display_height*0.2)))

    def reset(self):
        self.x = random.randrange(0, self.display_width)
        self.y = 0 - self.height

    def update(self):
        self.y += self.speed

    #def draw(self):
    #    pygame.draw.rect(self.gameDisplay, self.red, [self.x, self.y, self.width, self.height])

    def draw(self):
        self.gameDisplay.blit(self.img, (self.x,self.y))