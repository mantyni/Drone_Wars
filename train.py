import pygame 
import torch
import argparse

from DQN_agent import DQN_agent
from environment import DroneWars


def train():  
    pygame.init()
    clock = pygame.time.Clock()
    flags = pygame.SHOWN
    gameDisplay = pygame.display.set_mode((800,600), flags) 

    # Initiatlize the environment and initial observation
    env = DroneWars(gameDisplay, display_width=800, display_height=600, clock=clock, fps=200, num_drones=2, num_obstacles=2)
    state, _, _, _ = env.step(actions=[0,0])
    #state, _, _, _ = env.step(action1=0, action2=0)
    state = torch.cat(tuple(state for _ in range(4)))[None, :, :, :] 

    num_drones = 2
    drone_list = []
    action_list = []
    num_iterations = 400000 
    iter = 0
    
    # Initiatlize DQN agents
    for n in range(num_drones):
        drone_list.append(DQN_agent(id=n, train_mode=True, total_iterations=num_iterations))

    # Train the agents
    while iter < num_iterations:
        iter +=1 
        
        for drn in drone_list:
            action_list.append(drn.predict(state, train=True))
        
        #next_state, reward_list, done_list, info = env.step(action1 = action_list[0], action2 = action_list[1])   
        next_state, reward_list, done_list, info = env.step(actions=action_list)   
        next_state = torch.cat((state[0, 1:, :, :], next_state))[None, :, :, :]

        for drn in drone_list:
            drn.score = env.score
            
            drn.train(current_state=state, action=action_list[drn.id], next_state=next_state, reward=reward_list[drn.id], done=done_list[drn.id])
            
        state = next_state
        action_list = []

    print("Done training")


if __name__ == "__main__":
    train()

