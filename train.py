import argparse
import os
from random import random, randint, sample
import pickle
import numpy as np
import torch
import torch.nn as nn

from nn_model import DeepQNetwork
#from game import DroneWars
from game import *
from environment import DroneWars


def get_args():
    parser = argparse.ArgumentParser(
        """Reinforcement Learning Deep Q Network""")
    parser.add_argument("--batch_size", type=int, default=8, help="The number of images per batch") # was 64
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="adam")
    #parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=0.05)
    #parser.add_argument("--initial_epsilon", type=float, default=1e-3)
    parser.add_argument("--final_epsilon", type=float, default=1e-5)
    #parser.add_argument("--final_epsilon", type=float, default=1e-2)
    parser.add_argument("--num_decay_iters", type=float, default=1500000) #was 1000000
    parser.add_argument("--num_iters", type=int, default=1500000) # was 1000000
    # Replay memory size must not exeed available RAM, otherwise will crash
    # 10000 = 1Gb
    parser.add_argument("--replay_memory_size", type=int, default=1250, 
                        help="Number of epoches between testing phases") # was 20000
    parser.add_argument("--saved_folder", type=str, default="model")

    args = parser.parse_args()
    return args


def train(opt):

    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    model = DeepQNetwork()

    if torch.cuda.is_available():
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    if not os.path.isdir(opt.saved_folder):
        os.makedirs(opt.saved_folder)
    checkpoint_path = os.path.join(opt.saved_folder, "drone_wars.pth")
    memory_path = os.path.join(opt.saved_folder, "replay_memory.pkl")

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        iter = checkpoint["iter"] + 1
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("Load trained model from iteration {}".format(iter))
    else:
        iter = 0
    
    if os.path.isfile(memory_path):
        with open(memory_path, "rb") as f:
            replay_memory = pickle.load(f)
        print("Load replay memory")
    else:
        replay_memory = []
    criterion = nn.MSELoss()
    #env = DroneWars()
    env = DroneWars(gameDisplay, display_width, display_height, clock, fps)

    state, _, _ = env.step(0)
    state = torch.cat(tuple(state for _ in range(4)))[None, :, :, :]

    # Adding score
    
    all_scores = np.array(1)
    """
    action_dict = {
        (0,0) : [1,0,0,0,0,0,0,0,0],
        (0,1) : [0,1,0,0,0,0,0,0,0], 
        (0,2) : [0,0,1,0,0,0,0,0,0],
        (1,1) : [0,0,0,1,0,0,0,0,0],
        (1,0) : [0,0,0,0,1,0,0,0,0], 
        (1,2) : [0,0,0,0,0,1,0,0,0],
        (2,2) : [0,0,0,0,0,0,1,0,0],
        (2,0) : [0,0,0,0,0,0,0,1,0],
        (2,1) : [0,0,0,0,0,0,0,0,1] 
    }
    a = np.eye(9, dtype=int)
    actions = {}
    for n in range(9):
        actions[n] = a[n]
    """
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
    
    epsilon = 0.1

    while iter < opt.num_iters:

        # learning rate update:
        
        #if iter % 80000 == 0:
        #    optimizer.param_groups[0]['lr'] /= 10
        #    epsilon -= 0.02

        #if epsilon <= 0.01:
        #    epsilon = 0.01
        
        #if optimizer.param_groups[0]['lr'] <= 1e-5:
        #    optimizer.param_groups[0]['lr'] = 1e-5 

        if torch.cuda.is_available():
            prediction = model(state.cuda())[0]
        else:
            prediction = model(state)[0]
        # Exploration or exploitation
        #epsilon = opt.final_epsilon + (
        #        max(opt.num_decay_iters - iter, 0) * (opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_iters)
        
        epsilon = 0.001
        u = random.random()
        
        #epsilon = 0.01 # new hardcoding epsilon

        random_action = u <= epsilon
        #print("episolon: ", epsilon)
        #action = [0,0] # initialising empty actions
        if random_action:
            print("random")
            #action = randint(0, 2)
            action = randint(0, 8)
            #action[0] = randint(0, 2)
            #action[1] = randint(0, 2)
        else:
            print("normal")
            action = torch.argmax(prediction).item()
            #action[0] = torch.argmax(prediction[0:3]).item()
            #action[1] = torch.argmax(prediction[3:6]).item()

        #action = tuple(action)

        #print("Test pred, action ", prediction, torch.argmax(prediction).item())

        next_state, reward, done = env.step(action)
        next_state = torch.cat((state[0, 1:, :, :], next_state))[None, :, :, :]
        replay_memory.append([state, action, reward, next_state, done])
        
        if len(replay_memory) > opt.replay_memory_size:
            del replay_memory[0]
        
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = torch.cat(tuple(state for state in state_batch))

        #print("Action batch before: ", action_batch)
        # original
        #action_batch = torch.from_numpy(
        #    np.array([[1, 0, 0] if action == 0 else [0, 1, 0] if action == 1 else [0, 0, 1] for action in
        #              action_batch], dtype=np.float32))

        #print("Length of action_batch = ", len(action_batch))
        #print(action_batch)

        # new here
        """
        [1,0,0] if 0
        [0,1,0] if 1
        [0,0,1] if 2

        [1,0,0,1,0,0] if 0 0
        [0,1,0,1,0,0] if 1 0
        [1,0,0,0,1,0] if 0 1
        [0,1,0,0,1,0] if 1 1
        [0,1,0,0,0,1] if 1 2
        [0,0,1,0,0,1] if 2 2
        [0,0,1,0,1,0] if 2 1
        [1,0,0,0,0,1] if 0 2
        [0,0,1,1,0,0] if 2 0

        [1,0,0,0,0,0,0,0,0] if 0 0
        [0,1,0,0,0,0,0,0,0] if 0 1
        [0,0,1,0,0,0,0,0,0] if 0 2
        [0,0,0,1,0,0,0,0,0] if 1 1
        [0,0,0,0,1,0,0,0,0] if 1 0
        [0,0,0,0,0,1,0,0,0] if 1 2
        [0,0,0,0,0,0,1,0,0] if 2 2
        [0,0,0,0,0,0,0,1,0] if 2 0
        [0,0,0,0,0,0,0,0,1] if 2 1
        """

      

        #action_batch = torch.from_numpy(np.array(action_dict[action], dtype=np.float32))

        # New working for 2 drones:
        action_batch = torch.from_numpy(np.array([action_dict[action] for action in action_batch], dtype=np.float32))



        #print("New action batch: ", action_batch)
        # original: 
        #action_batch = torch.from_numpy(
        #    np.array([[1, 0, 0] if action == 0 else [0, 1, 0] if action == 1 else [0, 0, 1] for action in
        #              action_batch], dtype=np.float32))

        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.cat(tuple(state for state in next_state_batch))

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()
        
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

        # Keeping score list
        score = env.score
        all_scores = np.append(all_scores, score)


        state = next_state
        iter += 1

        #optimizer.param_groups[0]['lr']
        #print("Iteration: {}/{}, Loss: {:.5f}, Epsilon {:.5f}, Reward: {}, Score: {}".format(
        print("Iteration: {}/{}, Loss: {:.5f}, LR {:.5f}, Epsilon {:.5f}, Reward: {}, Score: {}".format(
            iter + 1,
            opt.num_iters,
            loss,
            optimizer.param_groups[0]['lr'], epsilon, reward, score))
        

        if (iter + 1) % 1250 == 0:
            print("Iteration: {}/{}, Loss: {:.5f}, Epsilon {:.5f}, Reward: {}, Score: {}".format(
            iter + 1,
            opt.num_iters,
            loss,
            epsilon, reward, score))
            
            checkpoint = {"iter": iter,
                          "model_state_dict": model.state_dict(),
                          "optimizer": optimizer.state_dict()}
            torch.save(checkpoint, checkpoint_path)
            
            print("# Saving model. Average Score: ", np.mean(all_scores))
            all_scores = np.array(1) # Reset all_scores list

            with open(memory_path, "wb") as f:
                pickle.dump(replay_memory, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    opt = get_args()
    print("Opt", opt)
    train(opt)
