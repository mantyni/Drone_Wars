import torch
import copy
import os
import pickle
import random
import numpy as np

from DQN_network import DQN_network

from torch.utils.tensorboard import SummaryWriter
# run tensorboard --logdir=runs
# open localhost:6006 in browser to see tensorboard


writer = SummaryWriter('runs/drone_wars')

class DQN_agent():
    
    def __init__(self, id=0, train_mode=False, total_iterations = 300000): 
        self.id = id
        self.train_mode = train_mode
        self.num_actions = 3
        self.batch_size = 32
        self.lr = 1e-4
        self.gamma = 0.99
        self.replay_memory = []
        self.replay_memory_size = 10000
        self.total_iterations = total_iterations
        self.update_starts = 5000 # Iterations to wait before target network updates starts (collect replay buffer)
        self.model_update_episodes = 10 # update traget network every x episodes
        self.model_save_rate = 20000 # save model every x iterations
        self.print_logs_episodes = 3 # print logs every x episodes
        self.main_network = DQN_network() # Main network - evaluation policy
        self.target_network = copy.deepcopy(self.main_network) # Target network - behavior policy
        self.save_folder = 'model'
        self.save_model_name = 'agent_' + str(id) + '.pth'
        self.save_replay_mem_name = 'agent_' + str(id) + '_replay_memory.pkl'
        self.criterion = torch.nn.MSELoss()
        self.episodes = 0
        self.network_update = False
        self.log_update = False
        self.score = 0
        self.rewards = []
        self.returns = []
        self.all_scores = np.array(1)
        self.top_score = 0
        self.loss = 0
        
        # Construct action dictionary
        a = np.eye(self.num_actions, dtype=int)
        self.action_dict = {}
        for n in range(self.num_actions):
            self.action_dict[n] = a[n]

        if torch.cuda.is_available():
            torch.cuda.manual_seed(123)
            self.device = 'cuda'
            self.main_network.cuda()
            self.target_network.cuda()
        else:
            torch.manual_seed(123)
            self.device = 'cpu'
        
        if not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder)
            
        self.checkpoint_path = os.path.join(self.save_folder, self.save_model_name)
        self.memory_path = os.path.join(self.save_folder, self.save_replay_mem_name)    

        # Initialize optimizer and set learning rate
        self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr=self.lr)
        
        # Load model 
        if os.path.isfile(self.checkpoint_path):
            print("Found checkpoint at: ", self.checkpoint_path)
            self.checkpoint = torch.load(self.checkpoint_path, map_location=torch.device(self.device))
            self.iter = self.checkpoint["iter"] + 1
            self.main_network.load_state_dict(self.checkpoint["model_state_dict"])
            self.target_network.load_state_dict(self.checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(self.checkpoint["optimizer"])
            print(f"Agent {self.id} loaded trained model from iteration {self.iter}")
            print(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}") # Model will load the last learning rate used!
        else:
            self.iter = 0
            print("Started training new model")

        if self.train_mode == True:
            if os.path.isfile(self.memory_path):
                with open(self.memory_path, "rb") as f:
                    self.replay_memory = pickle.load(f)
                print("Loaded replay memory of size: ", len(self.replay_memory))
            else:
                self.replay_memory = []
        
    def print_logs(self):
            print()
            print(f"###########")
            print(f"Agent: {self.id}")
            if self.iter < self.update_starts:
                print(f'Status: Pretraining')
            else:
                print(f'Status: Training')
            print(f"Step: {self.iter+1} / {self.total_iterations}")
            print(f"Episodes: {self.episodes}")
            print(f"Loss: {self.loss:.5f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"Mean Episode Return: {np.mean(self.returns):.4f}")
            print(f"Mean Episode Score: {np.mean(self.all_scores):.4f}")
            print(f"Top score: {self.top_score}")
            print(f"###########")
            print()

            self.log_update = True
            self.returns = []
            self.scores = []
            
    def update_target_network(self):         
            self.target_network = copy.deepcopy(self.main_network)
            self.network_update = True
            print(f"\n ### Agent {self.id} is updating target network### \n")
    
    def save_model(self):
            self.checkpoint = {"iter": self.iter, "model_state_dict": self.main_network.state_dict(), "optimizer": self.optimizer.state_dict()}
            torch.save(self.checkpoint, self.checkpoint_path)
            print(f"\n ## Agent {self.id} is saving model. Average Score: {np.mean(self.all_scores)}")
            self.all_scores = np.array(1) # Reset all_scores list

            with open(self.memory_path, "wb") as f:
                pickle.dump(self.replay_memory, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def predict(self, observation, train=False):
        if torch.cuda.is_available():
            observation = observation.cuda()
        
        if train == True:      
            if self.iter < self.update_starts:
                epsilon = 0.95
            else:
                epsilon = 0.01
                
            u = random.random()
            random_action = u <= epsilon
            
            if random_action:
                return random.randint(0,2)
                    
        if torch.cuda.is_available():
            with torch.no_grad(): 
                prediction = self.target_network(observation.cuda())[0]
        else:
            with torch.no_grad(): 
                prediction = self.target_network(observation)[0]
                
        return torch.argmax(prediction).item()
       
    def train(self, current_state, action, next_state, reward, done):
        if torch.cuda.is_available():
            current_state = current_state.cuda()
            next_state = next_state.cuda()
        
        #self.optimizer.param_groups[0]['lr'] = 1e-6

        # Update learning rate at iterations 150k and 250k (found to work through trial and error)
        if self.iter == 150000 or self.iter == 250000:
            self.lr = self.lr * 0.1
            self.optimizer.param_groups[0]['lr'] = self.lr

        self.replay_memory.append([current_state, action, reward, next_state, done])
        
        if len(self.replay_memory) > self.replay_memory_size:
            del self.replay_memory[0]
            
        batch = random.sample(self.replay_memory, min(len(self.replay_memory), self.batch_size))
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.cat(tuple(state for state in state_batch))
        action_batch = torch.from_numpy(np.array([self.action_dict[action] for action in action_batch], dtype=np.float32))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.cat(tuple(state for state in next_state_batch))
        
        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()
            
        with torch.no_grad(): 
            next_prediction_batch = self.target_network(next_state_batch)
            
        current_prediction_batch = self.main_network(state_batch)
        
        y_batch = torch.cat(tuple(reward if done else reward + self.gamma * torch.max(prediction) for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))
                       
        q_value = torch.sum(current_prediction_batch * action_batch, dim=1)

        self.optimizer.zero_grad()
        loss = self.criterion(q_value, y_batch)
        loss.backward()
        self.loss = loss.item() # Save loss as global variable for logs
        max_grad_norm = 1.0
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), max_grad_norm) # clip gradients to 10.0 TODO: review if it is needed to clip for Q network
        self.optimizer.step()

        self.iter += 1
        self.all_scores = np.append(self.all_scores, self.score)
        self.rewards.append(reward)
        if self.score > self.top_score:
            self.top_score = self.score

        # Episode is done
        if done:
            self.episodes += 1
            self.network_update = False
            self.log_update = False
            self.returns.append(np.sum(self.rewards)) # mean of returns of each episode
            self.rewards = []

        # Update target network
        if (self.iter > self.update_starts) and (self.episodes % self.model_update_episodes == 0) and (self.network_update == False):
            self.update_target_network()   

        # Save model every 20k iterations
        if (self.iter + 1) % self.model_save_rate == 0:
            self.save_model()
        
        # Print logs
        if (self.episodes % self.print_logs_episodes == 0) and (self.log_update == False):
            self.print_logs()
            
        # Tensorboard logs
        writer.add_scalar('Training loss Agent ' + str(self.id), loss.item(), self.iter)
        writer.add_scalar('Rewards Agent ' + str(self.id), reward, self.iter) 
        writer.add_scalar('Score Agent ' + str(self.id), self.score, self.iter) 