import argparse
import os
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import pygame
from nn_model import DeepQNetwork
from environment import DroneWars
from stable_baselines3.common.utils import polyak_update


def get_args():
    parser = argparse.ArgumentParser(
        """Reinforcement Learning Deep Q Network""")
    parser.add_argument("--batch_size", type=int, default=32, help="The number of images per batch") # was 64 # start at 8 increase up to 64 throught
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    #parser.add_argument("--lr", type=float, default=1e-4) #orig
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    #parser.add_argument("--initial_epsilon", type=float, default=1e-3)
    parser.add_argument("--final_epsilon", type=float, default=0.001) # was 1e-5
    #parser.add_argument("--final_epsilon", type=float, default=1e-2)
    parser.add_argument("--num_decay_iters", type=float, default=500000) #was 1000000
    parser.add_argument("--num_iters", type=int, default=650000) # was 1000000
    # Replay memory size must not exeed available RAM, otherwise will crash
    # 10000 = 1Gb
    parser.add_argument("--replay_memory_size", type=int, default=100000, 
                        help="Number of epoches between testing phases") # was 20000
    parser.add_argument("--saved_folder", type=str, default="model")
    parser.add_argument("--render", type=bool, default=True) 

    args = parser.parse_args()
    return args


def train(opt):
    
    pygame.init()
    clock = pygame.time.Clock()
    flags = pygame.SHOWN
    gameDisplay = pygame.display.set_mode((800,600), flags) 
    tau = 1
    update_starts = 1
    updated = False
    model_update_rate = 5 # number of episodes
    episodes = 0
    rewards = []
    scores = []
    all_scores = np.array(1)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    model = DeepQNetwork() # need to add a target network, initialy it's a copy # read more https://blog.gofynd.com/building-a-deep-q-network-in-pytorch-fa1086aa5435
                            # update every 5 steps

    model_target = DeepQNetwork()

    if torch.cuda.is_available():
        model.cuda()
        model_target.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    optimizer_target = torch.optim.Adam(model_target.parameters(), lr=opt.lr)

    # maybe redundant:
    optimizer.param_groups[0]['lr'] = opt.lr
    #optimizer_target.param_groups[0]['lr'] = opt.lr

    if not os.path.isdir(opt.saved_folder):
        os.makedirs(opt.saved_folder)
    checkpoint_path = os.path.join(opt.saved_folder, "drone_wars.pth")
    checkpoint_path_target = os.path.join(opt.saved_folder, "drone_wars_target.pth")
    memory_path = os.path.join(opt.saved_folder, "replay_memory.pkl")

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        checkpoint_target = torch.load(checkpoint_path_target)
        iter = checkpoint["iter"] + 1
        model.load_state_dict(checkpoint["model_state_dict"])
        model_target.load_state_dict(checkpoint["model_state_dict"])
        #model_target.load_state_dict(checkpoint_target["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        #optimizer_target.load_state_dict(checkpoint_target["optimizer"])
        print("Load trained model from iteration {}".format(iter))
    else:
        iter = 0
    
    if os.path.isfile(memory_path):
        with open(memory_path, "rb") as f:
            replay_memory = pickle.load(f)
        print("Load replay memory")
    else:
        replay_memory = []
    #criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss() # stablebaselines dqn is using huber loss

    env = DroneWars(gameDisplay, display_width=800, display_height=600, clock=clock, fps=200)

    state, _, _, _ = env.step(0)
    state = torch.cat(tuple(state for _ in range(4)))[None, :, :, :] # coppies same state over for 4 times
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
    while iter < opt.num_iters:

        if torch.cuda.is_available():
            #prediction = model(state.cuda())[0]
            if iter > update_starts:
                prediction = model_target(state.cuda())[0]
            else:
                prediction = model(state.cuda())[0]
        else:
            #prediction = model(state)[0]
            #prediction = model_target(state)[0]
            if iter > update_starts:
                prediction = model_target(state)[0]
            else:
                prediction = model(state)[0]
                
        # Exploration or exploitation
        epsilon = opt.final_epsilon + (
                max(opt.num_decay_iters - iter, 0) * (opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_iters)
        
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
        
        #current_prediction_batch = model(state_batch)
        #next_prediction_batch = model(next_state_batch)

        if iter > update_starts: 
            current_prediction_batch = model_target(state_batch)
            next_prediction_batch = model_target(next_state_batch)
        else: 
            current_prediction_batch = model(state_batch)
            next_prediction_batch = model(next_state_batch)

        y_batch = torch.cat(
            tuple(reward if done else reward + opt.gamma * torch.max(prediction) for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))

        q_value = torch.sum(current_prediction_batch * action_batch, dim=1)
        optimizer.zero_grad()
        loss = criterion(q_value, y_batch)
        loss.backward()
        optimizer.step()

        state = next_state
        
        # Keeping score list
        score = env.score
        iter += 1
        rewards.append(reward)
        scores.append(score)
        all_scores = np.append(all_scores, score)

        
        # Increment episode if done
        if done:
            episodes += 1
            updated = False
 
        # Update target network
        if (iter > update_starts) and ((episodes) % model_update_rate == 0) and (updated == False): #problem keeps updating until next episode 
            polyak_update(model.parameters(), model_target.parameters(), tau)
            updated = True
            
            
            print("###############################")
            print("### Updating target network ###")
            print("###############################")
            
            print(f"Episode: {episodes}")
            print(f"Step: {iter+1}/{opt.num_iters}")
            print(f"Loss: {loss:.5f}")
            print(f"LR: {optimizer.param_groups[0]['lr']:.5f}")
            print(f"Epsilon: {epsilon:.4f}")
            print(f"Mean Reward: {np.mean(rewards):.4f}")
            print(f"Mean Score: {np.mean(scores):.4f}")
            print()
            
            rewards = []
            scores = []

        # Save model
        if (iter + 1) % 5000 == 0:
            checkpoint = {"iter": iter,
                          "model_state_dict": model.state_dict(),
                          "optimizer": optimizer.state_dict()}
            torch.save(checkpoint, checkpoint_path)

            checkpoint_target = {iter: iter, 
                            "model_state_dict": model_target.state_dict(),
                            "optimizer": optimizer_target.state_dict()}
            torch.save(checkpoint_target, checkpoint_path_target)

            print("## Saving model. Average Score: ", np.mean(all_scores))
            all_scores = np.array(1) # Reset all_scores list

            with open(memory_path, "wb") as f:
                pickle.dump(replay_memory, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    opt = get_args()
    print("Opt", opt)
    train(opt)
