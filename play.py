import argparse
import torch
import pygame 
from nn_model import DeepQNetwork
from environment import DroneWars
import cv2

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
    parser.add_argument("--saved_path", type=str, default="./model")
    parser.add_argument("--fps", type=int, default=60, help="frames per second")
    parser.add_argument("--output", type=str, default="./output/drone_wars.mp4", help="the path to output video")

    args = parser.parse_args()
    return args


def play(opt):

    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    model1 = DeepQNetwork()
    checkpoint_path1 = "{}/drone_wars1.pth".format(opt.saved_path)
    
    model2 = DeepQNetwork()
    checkpoint_path2 = "{}/drone_wars2.pth".format(opt.saved_path)

    if torch.cuda.is_available():
        checkpoint1 = torch.load(checkpoint_path1)
        checkpoint2 = torch.load(checkpoint_path2)
    else:
        checkpoint1 = torch.load(checkpoint_path1, map_location=torch.device('cpu'))
        checkpoint2 = torch.load(checkpoint_path2, map_location=torch.device('cpu'))

        
    model1.load_state_dict(checkpoint1["model_state_dict"])
    model1.eval()

    model2.load_state_dict(checkpoint2["model_state_dict"])
    model2.eval()

    env = DroneWars(gameDisplay, display_width=800, display_height=600, clock=clock, fps=opt.fps, num_drones=2, num_obstacles=2)
    state, raw_state, _, _, _ = env.step(0, 0, True)
    state = torch.cat(tuple(state for _ in range(4)))[None, :, :, :]
    
    if torch.cuda.is_available():
        model1.cuda()
        model2.cuda()
        state = state.cuda()
        
    out = cv2.VideoWriter(opt.output, cv2.VideoWriter_fourcc(*"MJPG"), opt.fps, (800, 600))
    done = [False, False]
    
    while not done[0] or not done[1]:

        prediction1 = model1(state)[0]
        action1 = torch.argmax(prediction1).item()

        prediction2 = model2(state)[0]
        action2 = torch.argmax(prediction2).item()

        next_state, raw_next_state, reward, done, info = env.step(action1, action2, True)
        out.write(raw_next_state)
        
        if torch.cuda.is_available():
            next_state = next_state.cuda()
            
        next_state = torch.cat((state[0, 1:, :, :], next_state))[None, :, :, :]
        state = next_state


if __name__ == "__main__":
    opt = get_args()
    print(f"Parameters: {' '.join(f'{k}={v}' for k, v in vars(opt).items())}")
    play(opt)
