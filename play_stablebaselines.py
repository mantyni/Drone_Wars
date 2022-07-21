import pygame
from environment import DroneWars
from stable_baselines3 import DQN

import cv2

pygame.init()
clock = pygame.time.Clock()
flags = pygame.SHOWN
gameDisplay = pygame.display.set_mode((800,600), flags) 
fps = 60
env = DroneWars(gameDisplay, display_width=800, display_height=600, clock=clock, fps=fps) # 200

model = DQN.load("dqn_dronewars")

obs = env.reset()

steps = 1

out = cv2.VideoWriter("output/drone_wars.mp4", cv2.VideoWriter_fourcc(*"MJPG"), 30, (800, 600))

while steps > 0:
    action, _states = model.predict(obs, deterministic=True)
    # obs, reward, done, info = env.step(action)
    
    # for playing and recording uncomment below
    obs, raw_next_state, reward, done, info = env.step(action, record=True)
    out.write(raw_next_state)
    
    env.render()
    if done:
      obs = env.reset()
      steps -= 1
    
