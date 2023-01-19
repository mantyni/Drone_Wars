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
    parser.add_argument("--num_iters", type=int, default=600000) 
    parser.add_argument("--replay_memory_size", type=int, default=10000, help="Size of the memory buffer") # Replay memory must not exeed available RAM, otherwise will crash, 10000 = 1Gb
    parser.add_argument("--saved_folder", type=str, default="model")
    parser.add_argument("--render", type=bool, default=True) 
    parser.add_argument("--num_drones", type=int, default=2) 
    parser.add_argument("--num_obstacles", type=int, default=2)
    parser.add_argument("--fps", type=int, default=200) 
    
    args = parser.parse_args()
    return args


def train(opt):
    
    pygame.init()
    clock = pygame.time.Clock()
    flags = pygame.SHOWN
    gameDisplay = pygame.display.set_mode((800,600), flags) 
    update_starts = 5000 # Steps when to start updating target network
    updated = False
    log_update = False
    model_update_rate = 10 # number of episodes
    episodes = 0
    returns1 = []
    returns2 = []
    rewards1 = []
    rewards2 = []
    scores = []

    all_scores = np.array(1)
    lr = opt.lr
    max_grad_norm = 1.0 # was 10.0
    top_score = 0

    ep_score = []
    ep_memory = []
    tau = 0.001

    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    model1 = DeepQNetwork() # Main network - evaluation policy
    model1_target = copy.deepcopy(model1) # Target network - behavior policy

    model2 = DeepQNetwork() # Main network - evaluation policy
    model2_target = copy.deepcopy(model2) # Target network - behavior policy

    if torch.cuda.is_available():
        model1.cuda()
        model1_target.cuda()
        model2.cuda()
        model2_target.cuda()

    optimizer1 = torch.optim.Adam(model1.parameters(), lr=opt.lr)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=opt.lr)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, step_size=1500000, gamma=0.1) # TODO Learning rate scheduler 


    if not os.path.isdir(opt.saved_folder):
        os.makedirs(opt.saved_folder)

    checkpoint1_path = os.path.join(opt.saved_folder, "drone_wars1.pth")
    memory1_path = os.path.join(opt.saved_folder, "replay_memory1.pkl")

    checkpoint2_path = os.path.join(opt.saved_folder, "drone_wars2.pth")
    memory2_path = os.path.join(opt.saved_folder, "replay_memory2.pkl")

    if os.path.isfile(checkpoint1_path):
        checkpoint1 = torch.load(checkpoint1_path)
        iter = checkpoint1["iter"] + 1
        model1.load_state_dict(checkpoint1["model_state_dict"])
        model1_target.load_state_dict(checkpoint1["model_state_dict"])
        optimizer1.load_state_dict(checkpoint1["optimizer"])

        checkpoint2 = torch.load(checkpoint2_path)
        model2.load_state_dict(checkpoint2["model_state_dict"])
        model2_target.load_state_dict(checkpoint2["model_state_dict"])
        optimizer2.load_state_dict(checkpoint2["optimizer"])

        print("Load trained model from iteration {}".format(iter))
    else:
        iter = 0
        print("Starting training new model")

    
    if os.path.isfile(memory1_path):
        with open(memory1_path, "rb") as f:
            replay_memory1 = pickle.load(f)
        print("Load replay memory 1")

        with open(memory2_path, "rb") as f:
            replay_memory2 = pickle.load(f)
        print("Load replay memory 2")

    else:
        replay_memory1 = []
        replay_memory2 = []

    criterion = nn.MSELoss()
    #criterion = nn.SmoothL1Loss() # stablebaselines dqn is using huber loss

    env = DroneWars(gameDisplay, display_width=800, display_height=600, clock=clock, fps=opt.fps, num_drones=opt.num_drones, num_obstacles=opt.num_obstacles)

    state, _, _, _ = env.step(action1 = 0, action2 = 0)
    state = torch.cat(tuple(state for _ in range(4)))[None, :, :, :] # copies same state over for 4 times
    # [None, :, :, :] doesnt do anything...
    # uses 4 channels, coppies sames state info 4 times? # can change in nn to 2 channels
    
    """
    a = np.eye(9, dtype=int)
    actions = {}
    for n in range(9):
        actions[n] = a[n]
    """
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
    

    # Training loop:
    while iter < opt.num_iters:

        if torch.cuda.is_available():
            #prediction = model(state.cuda())[0]
            prediction1 = model1_target(state.cuda())[0]
            prediction2 = model2_target(state.cuda())[0]
        else:
            #prediction = model(state)[0]
            prediction1 = model1_target(state)[0]
            prediction2 = model2_target(state)[0]

                
        # Epislon decay:
        #epsilon = opt.final_epsilon + (max(opt.num_decay_iters - iter, 0) * (opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_iters)
        
        # Large epsilon at the begining to force agent explore
        # For this environment exploration is not necessary, training with epsilon = 0.01 all the time works best
        if iter < update_starts:
            epsilon = 0.95
        else:
            epsilon = 0.01

        if iter == 150000 or iter == 250000:
            lr = lr * 0.1
            optimizer1.param_groups[0]['lr'] = lr
            optimizer2.param_groups[0]['lr'] = lr

        #optimizer1.param_groups[0]['lr'] = 1e-5
        #optimizer2.param_groups[0]['lr'] = 1e-5


        u1 = random.random()
        random_action1 = u1 <= epsilon

        u2 = random.random()
        random_action2 = u2 <= epsilon

        if random_action1:
            action1 = random.randint(0, 2)
        else:
            action1 = torch.argmax(prediction1).item()

        if random_action2:
            action2 = random.randint(0, 2)
        else:
            action2 = torch.argmax(prediction2).item()


        next_state, reward, done, _ = env.step(action1, action2)
        next_state = torch.cat((state[0, 1:, :, :], next_state))[None, :, :, :]

        replay_memory1.append([state, action1, reward[0], next_state, done[0]])
        replay_memory2.append([state, action2, reward[1], next_state, done[1]])

        
        
        if len(replay_memory1) > opt.replay_memory_size:
            del replay_memory1[0]

        if len(replay_memory2) > opt.replay_memory_size:
            del replay_memory2[0]

        batch1 = random.sample(replay_memory1, min(len(replay_memory1), opt.batch_size))
        state_batch1, action_batch1, reward_batch1, next_state_batch1, done_batch1 = zip(*batch1)
        state_batch1 = torch.cat(tuple(state1 for state1 in state_batch1))
        action_batch1 = torch.from_numpy(np.array([action_dict[action1] for action1 in action_batch1], dtype=np.float32))
        reward_batch1 = torch.from_numpy(np.array(reward_batch1, dtype=np.float32)[:, None])
        next_state_batch1 = torch.cat(tuple(state1 for state1 in next_state_batch1))

        batch2 = random.sample(replay_memory2, min(len(replay_memory2), opt.batch_size))
        state_batch2, action_batch2, reward_batch2, next_state_batch2, done_batch2 = zip(*batch2)
        state_batch2 = torch.cat(tuple(state2 for state2 in state_batch2))
        action_batch2 = torch.from_numpy(np.array([action_dict[action2] for action2 in action_batch2], dtype=np.float32))
        reward_batch2 = torch.from_numpy(np.array(reward_batch2, dtype=np.float32)[:, None])
        next_state_batch2 = torch.cat(tuple(state2 for state2 in next_state_batch2))

        if torch.cuda.is_available():
            state_batch1 = state_batch1.cuda()
            action_batch1 = action_batch1.cuda()
            reward_batch1 = reward_batch1.cuda()
            next_state_batch1 = next_state_batch1.cuda()
        
            state_batch2 = state_batch2.cuda()
            action_batch2 = action_batch2.cuda()
            reward_batch2 = reward_batch2.cuda()
            next_state_batch2 = next_state_batch2.cuda()


        with torch.no_grad(): 
            next_prediction_batch1 = model1_target(next_state_batch1)
            next_prediction_batch2 = model2_target(next_state_batch2)


        current_prediction_batch1 = model1(state_batch1)
        current_prediction_batch2 = model2(state_batch2)


        y_batch1 = torch.cat(tuple(reward1 if done1 else reward1 + opt.gamma * torch.max(prediction1) for reward1, done1, prediction1 in
                  zip(reward_batch1, done_batch1, next_prediction_batch1)))


        y_batch2 = torch.cat(tuple(reward2 if done2 else reward2 + opt.gamma * torch.max(prediction2) for reward2, done2, prediction2 in
                  zip(reward_batch2, done_batch2, next_prediction_batch2)))


        loss1 = 0
        q_value1 = torch.sum(current_prediction_batch1 * action_batch1, dim=1)
        optimizer1.zero_grad()
        loss1 = criterion(q_value1, y_batch1)
        loss1.backward()
        torch.nn.utils.clip_grad_norm_(model1.parameters(), max_grad_norm) # clip gradients to 10.0
        optimizer1.step()

        
        loss2 = 0
        q_value2 = torch.sum(current_prediction_batch2 * action_batch2, dim=1)       
        optimizer2.zero_grad()
        loss2 = criterion(q_value2, y_batch2)
        loss2.backward()
        torch.nn.utils.clip_grad_norm_(model2.parameters(), max_grad_norm) # clip gradients to 10.0
        optimizer2.step()
        

        state = next_state
        
        # Keeping score list
        score = env.score
        iter += 1
        rewards1.append(reward[0])
        rewards2.append(reward[1])
        scores.append(score)
        all_scores = np.append(all_scores, score)
        if score > top_score:
            top_score = score

        # Log to tensorboard
        writer.add_scalar('training loss1', loss1.item(), iter) 
        writer.add_scalar('training loss12', loss2.item(), iter) 
        writer.add_scalar('rewards1', reward[0], iter) 
        writer.add_scalar('rewards2', reward[1], iter) 

        writer.add_scalar('score', score, iter) 

        # episodic scores (not needed anymore)
        if not done:
            ep_score = copy.deepcopy(score)

        #print(f"Episode Score: {ep_score:.4f}")

        # Increment episode if done
        #if done:
        if done[0] or done[1]:
            episodes += 1
            updated = False
            log_update = False
            returns1.append(np.sum(rewards1)) # mean of returns of each episode
            returns2.append(np.sum(rewards2)) # mean of returns of each episode
            rewards1 = []
            rewards2 = []
            
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
            print(f"Loss1: {loss1:.5f}")
            print(f"Loss2: {loss2:.5f}")
            print(f"LR: {optimizer1.param_groups[0]['lr']:.6f}")
            print(f"Epsilon: {epsilon:.4f}")
            print(f"Mean Episode Return 1: {np.mean(returns1):.4f}")
            print(f"Mean Episode Return 2: {np.mean(returns2):.4f}")
            print(f"Mean Episode Score: {np.mean(scores):.4f}")
            print(f"Top score: {top_score}")
            print()

            log_update = True
            returns1 = []
            returns2 = []
            scores = []


        # Update target network based on 'model_update_rate'
        if (iter > update_starts) and ((episodes) % model_update_rate == 0) and (updated == False):
            
            #polyak_update(model.parameters(), model_target.parameters(), tau) # Use for soft copy of params
            model1_target = copy.deepcopy(model1) # Hard copy params from main network to target network
            model2_target = copy.deepcopy(model2) # Hard copy params from main network to target network
            updated = True
            print("\n ### Updating target network ### \n")

        # Save model every 20k iterations
        if (iter + 1) % 20000 == 0:
            checkpoint1 = {"iter": iter, "model_state_dict": model1.state_dict(), "optimizer": optimizer1.state_dict()}
            checkpoint2 = {"iter": iter, "model_state_dict": model2.state_dict(), "optimizer": optimizer2.state_dict()}
            torch.save(checkpoint1, checkpoint1_path)
            torch.save(checkpoint2, checkpoint2_path)
            print("## Saving model. Average Score: ", np.mean(all_scores))
            all_scores = np.array(1) # Reset all_scores list

            with open(memory1_path, "wb") as f:
                pickle.dump(replay_memory1, f, protocol=pickle.HIGHEST_PROTOCOL)

            with open(memory2_path, "wb") as f:
                pickle.dump(replay_memory2, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    opt = get_args()
    print(f"Parameters: {' '.join(f'{k}={v}' for k, v in vars(opt).items())}")
    train(opt)
