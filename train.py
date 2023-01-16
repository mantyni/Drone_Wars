# Working training for multiple drones

import argparse
import os
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import copy
import pygame
from nn_model import DeepQNetwork
from environment import DroneWars

from stable_baselines3.common.utils import polyak_update
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/drone_wars')
# run tensorboard --logdir=runs
# open localhost:6006 in browser to see tensorboard


def get_args():
    parser = argparse.ArgumentParser("""Reinforcement Learning Deep Q Network""")
    parser.add_argument("--batch_size", type=int, default=32, help="The number of images per batch") 
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-4) 
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1.0)
    parser.add_argument("--final_epsilon", type=float, default=1e-3) 
    parser.add_argument("--num_decay_iters", type=float, default=500000) 
    parser.add_argument("--num_iters", type=int, default=1000000) # Replay memory must not exeed available RAM, otherwise will crash, 10000 = 1Gb
    parser.add_argument("--replay_memory_size", type=int, default=10000, help="Size of the memory buffer") 
    parser.add_argument("--saved_folder", type=str, default="model")
    parser.add_argument("--render", type=bool, default=True) 

    args = parser.parse_args()
    return args


def train(opt):
    
    pygame.init()
    clock = pygame.time.Clock()
    flags = pygame.SHOWN
    gameDisplay = pygame.display.set_mode((800,600), flags) 
    update_starts = 1 # Steps when to start updating target network
    updated = False
    log_update = False
    model_update_rate = 5 # number of episodes
    episodes = 0
    returns = []
    rewards = []
    scores = []

    all_scores = np.array(1)
    lr = opt.lr
    max_grad_norm = 10.0
    top_score = 0

    ep_score = []
    ep_memory = []
    tau = 0.001

    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    model = DeepQNetwork() # Main network - evaluation policy
    model_target = copy.deepcopy(model) # Target network - behavior policy

    if torch.cuda.is_available():
        model.cuda()
        model_target.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, step_size=1500000, gamma=0.1) # TODO Learning rate scheduler 


    if not os.path.isdir(opt.saved_folder):
        os.makedirs(opt.saved_folder)

    checkpoint_path = os.path.join(opt.saved_folder, "drone_wars.pth")
    memory_path = os.path.join(opt.saved_folder, "replay_memory.pkl")

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        iter = checkpoint["iter"] + 1
        model.load_state_dict(checkpoint["model_state_dict"])
        model_target.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("Load trained model from iteration {}".format(iter))
    else:
        iter = 0
        print("Starting training new model")

    
    if os.path.isfile(memory_path):
        with open(memory_path, "rb") as f:
            replay_memory = pickle.load(f)
        print("Load replay memory")
    else:
        replay_memory = []
    criterion = nn.MSELoss()
    #criterion = nn.SmoothL1Loss() # stablebaselines dqn is using huber loss

    env = DroneWars(gameDisplay, display_width=800, display_height=600, clock=clock, fps=200, num_drones=2, num_obstacles=2)

    state, _, _, _ = env.step(0)
    state = torch.cat(tuple(state for _ in range(4)))[None, :, :, :] # copies same state over for 4 times
    # [None, :, :, :] doesnt do anything...
    # uses 4 channels, coppies sames state info 4 times? # can change in nn to 2 channels
    
    """
    a = np.eye(9, dtype=int)
    actions = {}
    for n in range(9):
        actions[n] = a[n]
    """
    
    # multiple drones
    action_dict = {
        0 : [1,0,0,0,0,0,0,0,0],
        1 : [0,1,0,0,0,0,0,0,0], 
        2 : [0,0,1,0,0,0,0,0,0],
        3 : [0,0,0,1,0,0,0,0,0],
        4 : [0,0,0,0,1,0,0,0,0], 
        5 : [0,0,0,0,0,1,0,0,0],
        6 : [0,0,0,0,0,0,1,0,0],
        7 : [0,0,0,0,0,0,0,1,0],
        8 : [0,0,0,0,0,0,0,0,1] 
    }
    """
    # one drone
    action_dict = {
        0 : [1,0,0],
        1 : [0,1,0], 
        2 : [0,0,1],
    }
    """

    # Training loop:
    while iter < opt.num_iters:

        if torch.cuda.is_available():
            #prediction = model(state.cuda())[0]
            prediction = model_target(state.cuda())[0]
        else:
            #prediction = model(state)[0]
            prediction = model_target(state)[0]
                
        # Epislon decay:
        #epsilon = opt.final_epsilon + (max(opt.num_decay_iters - iter, 0) * (opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_iters)
        
        # Large epsilon at the begining to force agent explore
        # For this environment exploration is not necessary, training with epsilon = 0.01 all the time works best
        if iter < update_starts:
            epsilon = 0.95
        else:
            epsilon = 0.01

        if iter == 100000 or iter == 300000:
            lr = lr * 0.1
            optimizer.param_groups[0]['lr'] = lr
       
        u = random.random()
        random_action = u <= epsilon

        if random_action:
            #action = random.randint(0, 2) # single drone
            action = random.randint(0, 8)
        else:
            action = torch.argmax(prediction).item()

        next_state, reward, done, _ = env.step(action)
        next_state = torch.cat((state[0, 1:, :, :], next_state))[None, :, :, :]

        replay_memory.append([state, action, reward, next_state, done])

        
        # uncomment for testing memory buffer with quality episodes
        """
        if iter < update_starts:
            replay_memory.append([state, action, reward, next_state, done])
        else:
            ep_memory.append([state, action, reward, next_state, done])
        """

        
        if len(replay_memory) > opt.replay_memory_size:
            del replay_memory[0]
        
        batch = random.sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.cat(tuple(state for state in state_batch))
        action_batch = torch.from_numpy(np.array([action_dict[action] for action in action_batch], dtype=np.float32))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.cat(tuple(state for state in next_state_batch))

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()
        
        """
        if iter < update_starts:
            current_prediction_batch = model(state_batch)
            next_prediction_batch = model(next_state_batch)
        else:
            #current_prediction_batch = model_target(state_batch)
            #next_prediction_batch = model_target(next_state_batch)
            current_prediction_batch = model(state_batch)
            #next_prediction_batch = model(next_state_batch)
            next_prediction_batch = model_target(next_state_batch) # next q values are calculated using target network
        """
        with torch.no_grad(): 
            next_prediction_batch = model_target(next_state_batch)

        current_prediction_batch = model(state_batch)

        y_batch = torch.cat(tuple(reward if done else reward + opt.gamma * torch.max(prediction) for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))

        q_value = torch.sum(current_prediction_batch * action_batch, dim=1)
        
        optimizer.zero_grad()
        loss = criterion(q_value, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) # clip gradients to 10.0
        optimizer.step()

        state = next_state
        
        # Keeping score list
        score = env.score
        iter += 1
        rewards.append(reward)
        scores.append(score)
        all_scores = np.append(all_scores, score)
        if score > top_score:
            top_score = score

        # Log to tensorboard
        writer.add_scalar('training loss', loss.item(), iter) 
        writer.add_scalar('score', score, iter) 
        writer.add_scalar('rewards', reward, iter) 

        # episodic scores (not needed anymore)
        if not done:
            ep_score = copy.deepcopy(score)

        #print(f"Episode Score: {ep_score:.4f}")

        # Increment episode if done
        if done:
            episodes += 1
            updated = False
            log_update = False
            returns.append(np.sum(rewards)) # mean of returns of each episode
            rewards = []
            
            """
            # uncomment for testing memory buffer with quality episodes
            if iter > update_starts:
                if ep_score >= 10:
                    print("* Adding to memory - len of ep_memory", len(ep_memory))
                    print("Length of replay_memory: ", len(replay_memory))
                    for s, a, r, n, d in ep_memory:
                        replay_memory.append([s,a,r,n,d])
            """
            ep_memory = []
            ep_score = 0

        # Print logs every few episodes
        if (episodes % 5 == 0 and log_update == False):
            print(f"Episode: {episodes}")
            if iter < update_starts:
                print(f"Status: Pre-training")
            else:
                print(f"Status: Training")
            print(f"Step: {iter+1}/{opt.num_iters}")
            print(f"Loss: {loss:.5f}")
            print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"Epsilon: {epsilon:.4f}")
            print(f"Mean Episode Return: {np.mean(returns):.4f}")
            print(f"Mean Episode Score: {np.mean(scores):.4f}")
            print(f"Top score: {top_score}")
            print()

            log_update = True
            returns = []
            scores = []


        # Update target network based on 'model_update_rate'
        if (iter > update_starts) and ((episodes) % model_update_rate == 0) and (updated == False):
            
            #polyak_update(model.parameters(), model_target.parameters(), tau) # Use for soft copy of params
            model_target = copy.deepcopy(model) # Hard copy params from main network to target network
            updated = True
            print("\n ### Updating target network ### \n")

        # Save model every 20k iterations
        if (iter + 1) % 20000 == 0:
            checkpoint = {"iter": iter, "model_state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            torch.save(checkpoint, checkpoint_path)
            print("## Saving model. Average Score: ", np.mean(all_scores))
            all_scores = np.array(1) # Reset all_scores list

            with open(memory_path, "wb") as f:
                pickle.dump(replay_memory, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    opt = get_args()
    print(f"Parameters: {' '.join(f'{k}={v}' for k, v in vars(opt).items())}")
    train(opt)
