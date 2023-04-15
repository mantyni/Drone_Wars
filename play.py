import pygame 
import torch
import cv2 
import argparse

from DQN_agent import DQN_agent
from environment import DroneWars

pygame.init()
clock = pygame.time.Clock()
flags = pygame.SHOWN
gameDisplay = pygame.display.set_mode((800,600), flags) 


def get_args():
    parser = argparse.ArgumentParser(
        """
        ################################################################
        ### Reinforcement Learning Deep Q Network Playing Drone Wars ###
        ################################################################
         """)
    parser.add_argument("--fps", type=int, default=60, help="frames per second")
    parser.add_argument("--output", type=str, default="./output/drone_wars.mp4", help="the path to output video")
    parser.add_argument("--record", type=bool, default=True, help="record output video")
    
    args = parser.parse_args()
    return args


def play(opt):
    num_drones = 2
    drone_list = []
    # Initiatlize environment
    env = DroneWars(gameDisplay, display_width=800, display_height=600, clock=clock, fps=opt.fps, num_drones=num_drones, num_obstacles=2)
    # Initialize drones
    for x in range(num_drones):
        drone_list.append(DQN_agent(id=x))
    
    state, _, _, _ = env.step(action1 = 0, action2 = 0)
    state = torch.cat(tuple(state for _ in range(4)))[None, :, :, :] 

    action_list = []
    done_list = [False, False]
        
    out = cv2.VideoWriter(opt.output, cv2.VideoWriter_fourcc(*"MJPG"), opt.fps, (800, 600))

    while not done_list[0] or not done_list[1]:
        
        for drn in drone_list:
            action_list.append(drn.predict(state, train=False))
                
        if opt.record == True:
            next_state, raw_next_state, reward, done_list, info = env.step(action1 = action_list[0], action2 = action_list[1], record=True)
            out.write(raw_next_state)
        else:
            next_state, reward, done_list, info = env.step(action1 = action_list[0], action2 = action_list[1])   

        next_state = torch.cat((state[0, 1:, :, :], next_state))[None, :, :, :]
        state = next_state
        action_list = []


if __name__ == "__main__":
    opt = get_args()
    print()
    print(f"Game parameters: {' '.join(f'{k}={v}' for k, v in vars(opt).items())}")
    print()
    play(opt)