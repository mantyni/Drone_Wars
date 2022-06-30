from stable_baselines3.common.env_checker import check_env

import pygame
from environment import DroneWars

from stable_baselines3 import DQN

pygame.init()
clock = pygame.time.Clock()
flags = pygame.SHOWN
gameDisplay = pygame.display.set_mode((800,600), flags) 
env = DroneWars(gameDisplay, display_width=800, display_height=600, clock=clock, fps=150)


check_env(env, warn=True)