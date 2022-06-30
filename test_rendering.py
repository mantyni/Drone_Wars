import pygame
import numpy as np
#from pygame.surfarray import array3d

pygame.init()

clock = pygame.time.Clock()

# Display parameters
display_width = 800
display_height = 600
fps = 30
#flags = pygame.HIDDEN
flags = pygame.SHOWN
gameDisplay = pygame.display.set_mode((display_width,display_height), flags) 

