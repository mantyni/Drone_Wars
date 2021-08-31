import pygame
import random

class Obstacle:
    def __init__(self, gameDisplay, display_width=800, *args, **kwargs):
        self.x = 0
        self.y = 0
        self.speed = 30
        self.display_width = display_width
        self.height = 100
        self.width = 100
        self.red = (255,0,0)
        self.gameDisplay = gameDisplay


    def reset(self):
        self.x = random.randrange(0, self.display_width)
        self.y = 0 - self.height

    def update(self):
        self.y += self.speed

    def draw(self):
        pygame.draw.rect(self.gameDisplay, self.red, [self.x, self.y, self.width, self.height])