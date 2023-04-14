import pygame 
import torch

from DQN_agent import DQN_agent
from environment import DroneWars


pygame.init()
clock = pygame.time.Clock()
flags = pygame.SHOWN
gameDisplay = pygame.display.set_mode((800,600), flags) 

env = DroneWars(gameDisplay, display_width=800, display_height=600, clock=clock, fps=200, num_drones=2, num_obstacles=2)

state, _, _, _ = env.step(action1 = 0, action2 = 0)
state = torch.cat(tuple(state for _ in range(4)))[None, :, :, :] 

num_drones = 2
drone_list = []
action_list = []

for n in range(num_drones):
    drone_list.append(DQN_agent(id=n))

num_iterations = 1000
iter = 0

while iter < num_iterations:
    iter +=1 
    
    for drn in drone_list:
        action_list.append(drn.predict(state, train=True))
       
    next_state, reward_list, done_list, info = env.step(action1 = action_list[0], action2 = action_list[1])   
    next_state = torch.cat((state[0, 1:, :, :], next_state))[None, :, :, :]

    for drn in drone_list:
        drn.score = env.score
        drn.train(current_state=state, action=action_list[0], next_state=next_state, reward=reward_list[0], done=done_list[0])
        
    state = next_state
    action_list = []

print("Done")