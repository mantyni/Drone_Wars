import pygame
from stable_baselines3 import DQN, PPO, A2C
import cv2

from environment import DroneWars


pygame.init()
clock = pygame.time.Clock()
flags = pygame.SHOWN
width = 800
height = 600
gameDisplay = pygame.display.set_mode((width,height), flags) 

env = DroneWars(gameDisplay, display_width=width, display_height=height, clock=clock, num_drones=2, num_obstacles=1, fps=200) # 200

# DQN
# Uncomment below for training:
#model = DQN("CnnPolicy", env, buffer_size=10000, verbose=2) 
#model.learn(total_timesteps=300000, log_interval=5)
#model = DQN("MlpPolicy", env, buffer_size=10000, verbose=1) # can use either mlp or cnn policy
#model.save("dqn_dronewars")
#model.save_replay_buffer("dqn_replay_buffer")

# Uncomment below for testing:
model = DQN.load("dqn_dronewars")
model.load_replay_buffer("dqn_replay_buffer")
print(f"The loaded_model has {model.replay_buffer.size()} transitions in its buffer")

"""
# PPO
model = A2C("CnnPolicy", env, verbose=2) # PPO
model.save("ppo_dronewars")
model.learn(total_timesteps=600000, log_interval=5)
model = A2C.load("ppo_dronewars")
#print(f"The loaded model {model} ")
"""

obs = env.reset()

episodes = 1 # Number of episodes to play after training

out = cv2.VideoWriter("output/drone_wars.mp4", cv2.VideoWriter_fourcc(*"MJPG"), 60, (width, height))

while episodes > 0:
    print("Playing")
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    
    env.render()

    if done:
      obs = env.reset()
      episodes -= 1
    
