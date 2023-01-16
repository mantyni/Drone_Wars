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
    parser.add_argument("--fps", type=int, default=100, help="frames per second")
    parser.add_argument("--output", type=str, default="./output/drone_wars.mp4", help="the path to output video")

    args = parser.parse_args()
    return args


def play(opt):

    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    model = DeepQNetwork()
    checkpoint_path = "{}/drone_wars.pth".format(opt.saved_path)
    
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    env = DroneWars(gameDisplay, display_width=800, display_height=600, clock=clock, fps=opt.fps)
    state, raw_state, _, _, _ = env.step(0, True)
    state = torch.cat(tuple(state for _ in range(4)))[None, :, :, :]
    
    if torch.cuda.is_available():
        model.cuda()
        state = state.cuda()
        
    #out = cv2.VideoWriter(opt.output, cv2.VideoWriter_fourcc(*"MJPG"), opt.fps, (800, 600))
    done = False
    
    while not done:
        prediction = model(state)[0]
        action = torch.argmax(prediction).item()
        next_state, raw_next_state, reward, done, info = env.step(action, True)
        #out.write(raw_next_state)
        
        if torch.cuda.is_available():
            next_state = next_state.cuda()
            
        next_state = torch.cat((state[0, 1:, :, :], next_state))[None, :, :, :]
        state = next_state


if __name__ == "__main__":
    opt = get_args()
    print(f"Parameters: {' '.join(f'{k}={v}' for k, v in vars(opt).items())}")
    play(opt)
