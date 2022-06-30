import pygame
from drone import Drone
from obstacle import Obstacle
import numpy as np
np.set_printoptions(threshold=np.inf)
from pygame.surfarray import array3d
import torch
import cv2
from game import scoreboard

import gym 
from gym import spaces 


def pre_processing(image, w=84, h=84):
    image = image[:800, 20:, :] # crop out the top so score is not visible
    #cv2.imwrite("original.jpg", image)
    image = cv2.cvtColor(cv2.resize(image, (w, h)), cv2.COLOR_BGR2GRAY)
    #cv2.imwrite("color.jpg", image)
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    #cv2.imwrite("bw.jpg", image)

    a = np.array(image[None, :, :]).astype(np.float32) 
    #a = image[None, :, :].astype(np.uint8) # use for open ai baselines
    a = a / 255 # normalise the outputs 
    #print("TESTING")
    #print(a) 

    return a #image[None, :, :].astype(np.float32)


class DroneWars(gym.Env):
    def __init__(self, gameDisplay, display_width=800, display_height=600, clock=None, fps = 30, *args, **kwargs):
        super(DroneWars, self).__init__()
        self.my_drone1 = Drone(gameDisplay)
        self.my_drone1.x = display_width * 0.8
        self.my_drone1.y = 500 

        self.my_drone2 = Drone(gameDisplay)
        self.my_drone2.x = display_width * 0.2
        self.my_drone2.y = 500 # TODO, fix hardcoded y coordinate. Before was display_height * 0.5    

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
        self.obstacle_list = []
        self.n_actions = 9
        self.action_space = spaces.Discrete(self.n_actions)
        #self.observation_space = spaces.Box(low=0, high=255, shape=(1, 84, 84), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(1,84,84), dtype=np.uint8) #needed for cnn policy
        #self.observation_space = {
        #    'observation_space' : spaces.Box(low=0, high=1, shape=(1,84,84), dtype=np.float32)
        #}



        for n in range(0,1):
            self.obstacle_list.append(Obstacle(gameDisplay))
        pygame.display.set_caption('Drone Wars')


    def close(self):
        pass

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array) 
        """
        #r = np.zeros((1,84,84)).astype(np.float32)
        r = np.zeros((1,84,84)).astype(np.uint8)
        return r

    def render(self):

        self.gameDisplay.fill(self.white) # Comment this out if using scrolBackground
        for obs in self.obstacle_list:
            obs.draw()
            
        self.my_drone1.draw()
        self.my_drone2.draw()

        scoreboard(self.score)

        pygame.display.update()


    def scoreboard(self, count):

        font = pygame.font.SysFont(None, 25)
        text = font.render("Score: "+str(count), True, black)
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


    def step(self, action, record=False): # 0: do nothing, 1: go left, 2: go right
        
        #print("Testing obs space:", self.observation_space)
        #print("testing np array zeros", np.zeros((84,84)).astype(np.float32))
        reward = 0.1
        
        if action == 0:
            pass
            #print("Action: 0, do nothing")
            #reward += 0.01
            
        
        if action == 1:
            # drone1 do nothing
            # drone2 move left
            #print("Action: 1, drone2 left")
            self.my_drone2.move_left()
            
        if action == 2:
            #drone 1 do nothing
            #drone 2 move right
            #print("Action: 2, drone2 right")
            self.my_drone2.move_right()
        
        if action == 3:
            #drone 1 & 2 move left
            #print("Action: 3, drone1 left, drone2 move left")
            self.my_drone1.move_left()
            self.my_drone2.move_left()

        if action == 4:
            #drone 1 move left
            #drone 2 do nothing
            #print("Action: 4, drone1 left")
            self.my_drone1.move_left()

        if action == 5:
            #drone 1 move left 
            #drone 2 move right
            #print("Action: 3, drone1 left, drone2 move right")
            self.my_drone1.move_left()
            self.my_drone2.move_right()

        if action == 6:
            #drone 1&2 mvoe right
            #print("Action: 6, drone1 right, drone2 move right")
            self.my_drone1.move_right()
            self.my_drone2.move_right()

        if action == 7:
            #drone 1 move right
            #print("Action: 7, drone1 right")
            self.my_drone1.move_right()
            # drone 2 do nothing

        if action == 8:
            #print("Action: 8, drone1 right, drone2 move left")
            self.my_drone1.move_right()
            self.my_drone2.move_left()
            #drone 1 move right
            # drone 2 move left
        
        
        """
        0 do nothing
        1 move left
        2 move right

        0 [1,0,0,0,0,0,0,0,0] if 0 0
        1 [0,1,0,0,0,0,0,0,0] if 0 1
        2 [0,0,1,0,0,0,0,0,0] if 0 2
        3 [0,0,0,1,0,0,0,0,0] if 1 1
        4 [0,0,0,0,1,0,0,0,0] if 1 0
        5 [0,0,0,0,0,1,0,0,0] if 1 2
        6 [0,0,0,0,0,0,1,0,0] if 2 2
        7 [0,0,0,0,0,0,0,1,0] if 2 0
        8 [0,0,0,0,0,0,0,0,1] if 2 1
        """


        # single drone
        """
        if action == 0:
            pass
        #    reward += 0.01

        elif action == 1:
            self.my_drone1.move_left()

        elif action == 2:
            self.my_drone1.move_right()
        """
        
        # Update drone 1 & 2 position 
        self.my_drone1.update()
        self.my_drone2.update()

        # Update obstacle position. Move obstacle down the screen.
        #self.my_obstacle.update()
        for obs in self.obstacle_list:
            obs.update()



        # Detect if obstacle went to the bottom of the screen, then reset y & x coordinates to start from the top again at a random x coordinate. 
        # Increase obstacles speed as the game progresses. 
        for obs in self.obstacle_list:
            if obs.y > self.display_height:
                obs.reset()
                reward = 1
                self.score += 1

        #if self.my_obstacle.y > self.display_height:
        #    self.my_obstacle.reset()
        #    reward = 1
        #    self.score += 1
            #reward += 0.1
            #if self.my_obstacle.speed < 50:
            #    self.my_obstacle.speed = 1.15 * self.my_obstacle.speed

        # Detect if drone1 left the display bounds, then game over
        if self.out_of_bounds(self.my_drone1, self.display_width, self.display_height):
            #crash()
            reward = -1
            self.gameExit = True

        if self.out_of_bounds(self.my_drone2, self.display_width, self.display_height):
            #crash()
            reward = -1
            self.gameExit = True

        # Detect when obstacle collides with the drone1 and reduce the score 
        #if self.collision(self.my_drone1, self.my_obstacle):
        if self.collision_multi(self.my_drone1, self.obstacle_list):
            self.score -= 1 
            reward = -1
            self.gameExit = True

        # Detect when obstacle collides with the drone2 and reduce the score 
        #if self.collision(self.my_drone2, self.my_obstacle):
        if self.collision_multi(self.my_drone2, self.obstacle_list):
            self.score -= 1 
            reward = -1
            self.gameExit = True


        
        # Added 

        #self.gameDisplay.fill(self.white) # Comment this out if using scrolBackground
        #self.my_obstacle.draw()
        #for obs in self.obstacle_list:
        #    obs.draw()
            
        #self.my_drone1.draw()
        #self.my_drone2.draw()
        #scoreboard(self.score)

        #pygame.display.update()

        self.render()

        self.clock.tick(self.fps) 
        #print("clock:", self.clock.get_fps()) # actual fps
        #print("fps", self.fps) # set fps

        if self.gameExit:
            self.__init__(self.gameDisplay, self.display_width, self.display_height, self.clock, self.fps)
        
        state = pygame.display.get_surface() # should be at the bottom

        state = array3d(state)
       


        done = (not (reward > 0))
        info = {}

        if record:
            return torch.from_numpy(pre_processing(state)), np.transpose(
                cv2.cvtColor(state, cv2.COLOR_RGB2BGR), (1, 0, 2)), reward, not (reward > 0)
        else:
            return torch.from_numpy(pre_processing(state)), reward, done, info
            #return pre_processing(state), reward, done, info # use for gym baselines