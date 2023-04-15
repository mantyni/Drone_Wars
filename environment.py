import pygame
import torch
import cv2
import numpy as np
import gym 

from drone import Drone
from asteroid import Asteroid


def pre_processing(image, w=84, h=84, out=False):
    image_orig = image[:800, 20:, :] # crop out the top so score is not visible
    image_color = cv2.cvtColor(cv2.resize(image_orig, (w, h)), cv2.COLOR_BGR2GRAY)
    _, image_bw = cv2.threshold(image_color, 127, 255, cv2.THRESH_BINARY)
 
    if out == True:
        cv2.imwrite("original.jpg", image_orig) # save original screen
        cv2.imwrite("bw.jpg", image_bw) # save black and white image
        cv2.imwrite("color.jpg", image_color) # save colored image

    # Uncomment below for open ai baselines:
    #a = image_bw[None, :, :].astype(np.uint8) 
    
    # Uncomment below for custom network
    a = np.array(image_bw[None, :, :]).astype(np.float32) 
    a = a / 255 # normalise the outputs 

    return a 


class DroneWars(gym.Env):
    
    def __init__(self, gameDisplay, display_width=800, display_height=600, clock=None, fps = 30, num_drones = 2, num_obstacles = 2, *args, **kwargs):
        super().__init__()
        
        self.gameDisplay = gameDisplay
        self.display_width = display_width
        self.display_height = display_height
        self.score = 0
        self.gameExit = False
        self.clock = clock
        self.fps = fps
        self.black = (0,0,0)
        self.white = (255,255,255)
        self.dark_red = (150,0,0)
        self.green = (0,255,0)
        self.dark_green = (0,150,0)
        self.red = (255,0,0)
        self.n_actions = 3
        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1, 84, 84), dtype=np.float32) #use for custom NN without open baselines
        #self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1,84,84), dtype=np.uint8) #needed for cnn policy for open baselines
        self.num_of_obstacles = num_obstacles # nuber of obstacles
        self.num_of_drones = num_drones
        self.obstacle_list = []
        self.drone_list = []
        self.reward_list = [0,0]
        self.done_list = [False, False]

        # Construct drone and obstacle objects
        for i in range(self.num_of_drones):
            self.drone_list.append(Drone(gameDisplay, drone_id=i))

        for i in range(self.num_of_obstacles):
            self.obstacle_list.append(Asteroid(gameDisplay))

        pygame.display.set_caption('Drone Wars')
        
    def close(self):
        pass

    def reset(self):
        r = np.zeros((1,84,84)).astype(np.float32) # use for custom model
        #r = np.zeros((1,84,84)).astype(np.uint8) # use for openbaselines. TODO check why works fine for custom model
        return r

    def render(self):
        self.gameDisplay.fill(self.white) # Comment this out if using scrolBackground
        for obs in self.obstacle_list:
            obs.draw()

        for drn in self.drone_list:
            drn.draw()

        self.scoreboard(self.score)
        pygame.display.update()

    # not being used
    def scoreboard(self, count):
        self.black = (0,0,0)
        font = pygame.font.SysFont(None, 25)
        text = font.render("Score: "+str(count), True, self.black)
        self.gameDisplay.blit(text,(0,0))

    def out_of_bounds(self, drone, display_width, display_height):
        if (drone.x > display_width - drone.drone_width or drone.x < 0) or \
            (drone.y > display_height - drone.drone_height or drone.y < 0):
            
            return True 

    def collision_multi(self, drone, obstacle_list):
        for obs in obstacle_list:
            if (drone.y < obs.y + obs.height):

                if (drone.x > obs.x
                    and drone.x < obs.x + obs.width or drone.x + drone.drone_width > obs.x 
                    and drone.x + drone.drone_width < obs.x + obs.width):
                    
                    return True   

    def collision(self, drone, obstacle):
            if (drone.y < obstacle.y + obstacle.height):

                if (drone.x > obstacle.x
                    and drone.x < obstacle.x + obstacle.width or drone.x + drone.drone_width > obstacle.x 
                    and drone.x + drone.drone_width < obstacle.x + obstacle.width):
                    
                    return True   

    def step(self, action1, action2, record=False): # 0: do nothing, 1: go left, 2: go right

        self.reward_list = [0.1,0.1] # initialize reward for each drone
        
        for drn in self.drone_list:
            drn.reward = 0.1
        
        # Implement:
        #for drn in self.drone_list:
            #for a in self.actions:

        # For single drone actions in decentralized environment
        # first drone actions
        if action1 == 0:
            self.reward_list[0] += 0.01

        if action1 == 1:
            self.drone_list[0].move_left()       

        if action1 == 2:
            self.drone_list[0].move_right()       
        
        # second drone actions
        if action2 == 0:
            self.reward_list[1] += 0.01

        if action2 == 1:
            self.drone_list[1].move_left()       

        if action2 == 2:
            self.drone_list[1].move_right()       
        
        # Update drone 1 & 2 position 
        for drn in self.drone_list:
            drn.update()

        # Update obstacle position. Move obstacle down the screen.
        for obs in self.obstacle_list:
            obs.update()

        # Detect if obstacle went to the bottom of the screen, then reset y & x coordinates to start from the top again at a random x coordinate. 
        for obs in self.obstacle_list:
            if obs.y > self.display_height:
                obs.reset()
                self.reward_list[0] += 1 # should reward increment or be 1
                self.reward_list[1] += 1
                self.score += 1

        # Detect if drone1 left the display bounds, then game over
        for drn in self.drone_list:
            if self.out_of_bounds(drn, self.display_width, self.display_height):
                self.reward_list[drn.id] = -1
                self.gameExit = True

        # Detect when obstacle collides with the drone1 and reduce the score 
        for drn in self.drone_list:
            if self.collision_multi(drn, self.obstacle_list):
                self.score -= 1 
                self.reward_list[drn.id] = -1
                self.gameExit = True

        self.render()
        self.clock.tick(self.fps) 
        #print("clock:", self.clock.get_fps()) # Uncomment to printout actual fps 

        if self.gameExit:
            self.__init__(self.gameDisplay, self.display_width, self.display_height, self.clock, self.fps, self.num_of_drones, self.num_of_obstacles)
        
        state = pygame.display.get_surface() 
        state = pygame.surfarray.array3d(state)
        
        self.done_list[0] = (not (self.reward_list[0] > 0))
        self.done_list[1] = (not (self.reward_list[1] > 0))
        info = {}
        
        if record:
            return torch.from_numpy(pre_processing(state)), np.transpose(cv2.cvtColor(state, cv2.COLOR_RGB2BGR), (1, 0, 2)), self.reward_list, self.done_list, info  # custom training network
            #return pre_processing(state), np.transpose(cv2.cvtColor(state, cv2.COLOR_RGB2BGR), (1, 0, 2)), reward, done, info # Use for openbaselines
        else:
            return torch.from_numpy(pre_processing(state)), self.reward_list, self.done_list, info # Use for custom network training
            #return pre_processing(state), reward, done, info # use for gym baselines