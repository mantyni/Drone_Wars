import gym 
from gym import spaces 
import pygame
from environment import DroneWars
import numpy as np
from stable_baselines3 import DQN

import cv2

pygame.init()
clock = pygame.time.Clock()
flags = pygame.SHOWN
gameDisplay = pygame.display.set_mode((800,600), flags) 
fps = 60
env = DroneWars(gameDisplay, display_width=800, display_height=600, clock=clock, fps=fps) # 200

#model = DQN("MlpPolicy", env, buffer_size=10000, verbose=1) # can use either mlp or cnn policy

# uncomment for training:
"""
model = DQN("CnnPolicy", env, buffer_size=10000, verbose=2)

model.learn(total_timesteps=300000, log_interval=5)
model.save("dqn_dronewars")
"""
model = DQN.load("dqn_dronewars")

obs = env.reset()

steps = 1

out = cv2.VideoWriter("output/drone_wars.mp4", cv2.VideoWriter_fourcc(*"MJPG"), 30, (800, 600))

while steps > 0:
    print("Playing")
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    
    # for playing and recording uncomment below
    obs, raw_next_state, reward, done, info = env.step(action, record=True)
    out.write(raw_next_state)
    
    env.render()
    if done:
      obs = env.reset()
      steps -= 1
    
