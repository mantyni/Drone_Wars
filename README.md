# Drone Wars

## Introduction 
I made this game as a testbed platform for developing an AI agent that avoids asteroids and defeats bad robots. 

## First iteration, simple heuristic agent behaviour 
The agent follows predefined rules for avoiding incoming asteroids, however, does not reach high scores. 

![Alt text](images/gameplay.gif "Gameplay")


## Second iteration, Reinforcement Learning Deep Q method
The agent is trained with RL Deep Q method. The results are great, agent avoids asteroids at a very high game speed, better than human could. However, it is still not perfect and sometimes a miss happens. 


![Alt text](images/gameplay_rl.gif "Gameplay_RL")


## Third iteration, Reinforcement Learning with Double Deep Q method and multiple drones
The agent is trained to control 2 drones in a centralised fashion. The game problem complexity is much higher and it took significantly longer time to train the agent to reach reasonable score.
Implementation description will be released soon. 


![Alt text](images/gameplay_rl_2.gif "Gameplay_Multi_Agent_RL")

## Instructions 

There are two implementations to train Double Q Network for Drone Wars, one using Stable Baselines 3 and another custom implementation from scratch using Pytorch. Currently custom network implementation lags behind Stable Baselines3 performance in training speed and agent scores, however, its custom implementation is more transparent and easier to undestand. 

### Colab
Use Colab notebooks to train the models:
`drone_wars_custom_network.ipynb` - trains the model using custom Double Q network in Pytorch
`drone_wars_stable_baselines.ipynb` - trains the model using Stable Baselines 3 

To test the agent download use the scripts to play: 
`python3 play.py` - plays the agent with custom network model
`python3 play_stablebaselines` - plays the agent with Stable Baselines model

### Python scripts
Instead of Colab you can use Python scripts to train the models
`python3 train.py`
`python3 train_stablebaselines_dqn.py`

To play the game manually or with scripted AI run: 
`python3 game.py`

## Requirements
* python3
* pygame
* pygame-menu
* torch
* numpy
* stable_baselines3
* gym
* cv2


## Next development steps
* Improve the game 
    * The drone can shoot asteroids
    * Add top scores to the menu
    * Add different types of asteroids
    * Add power-ups for the drone
* Improve AI
    * Improve RL method to reach higher scores
* Multi-agent 
    * Make the game multi-agent so more than 1 drone is playing
    * Adapt RL method for multi-agent game setting





