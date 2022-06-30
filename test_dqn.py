import gym 
from gym import spaces 
import pygame
from environment import DroneWars
import numpy as np
from stable_baselines3 import DQN

pygame.init()
clock = pygame.time.Clock()
flags = pygame.SHOWN
gameDisplay = pygame.display.set_mode((800,600), flags) 

env = DroneWars(gameDisplay, display_width=800, display_height=600, clock=clock, fps=30)

#model = DQN("MlpPolicy", env, buffer_size=10000, verbose=1) # can use either mlp or cnn policy
model = DQN("CnnPolicy", env, buffer_size=10000, verbose=2)

model.learn(total_timesteps=200000, log_interval=5)
model.save("dqn_dronewars")

#del model # remove to demonstrate saving and loading

model = DQN.load("dqn_dronewars")
#print("Test", model.learning_rate)

obs = env.reset()

steps = 10

while steps > 0:
    print("Playing")
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()
      steps -= 1