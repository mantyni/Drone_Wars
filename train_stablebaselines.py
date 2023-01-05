import pygame
from environment import DroneWars
from stable_baselines3 import DQN
import cv2

pygame.init()
clock = pygame.time.Clock()
flags = pygame.SHOWN
width = 800
height = 600
gameDisplay = pygame.display.set_mode((width,height), flags) 

env = DroneWars(gameDisplay, display_width=width, display_height=height, clock=clock, fps=200) # 200

#model = DQN("MlpPolicy", env, buffer_size=10000, verbose=1) # can use either mlp or cnn policy

# Uncomment below for training:
model = DQN("CnnPolicy", env, buffer_size=10000, verbose=2)
model.learn(total_timesteps=300000, log_interval=5)
model.save("dqn_dronewars")
model.save_replay_buffer("dqn_replay_buffer")
# End of training code 

# Load trained model
model = DQN.load("dqn_dronewars")
model.load_replay_buffer("dqn_replay_buffer")
print(f"The loaded_model has {model.replay_buffer.size()} transitions in its buffer")

obs = env.reset()

episodes = 1 # Number of episodes to play after training

out = cv2.VideoWriter("output/drone_wars.mp4", cv2.VideoWriter_fourcc(*"MJPG"), 60, (width, height))

while episodes > 0:
    print("Playing")
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    
    # To record gameplay uncomment below
    #obs, raw_next_state, reward, done, info = env.step(action, record=True)
    #out.write(raw_next_state)
    
    env.render()
    if done:
      obs = env.reset()
      episodes -= 1
    
