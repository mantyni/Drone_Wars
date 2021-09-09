# Drone Wars

## Introduction 
I made this game as a testbed platform to develop an AI agebt that avoid asteroids and defeats bad robots. Currently AI is scripted and the drone agent does not reach very high score. Next I will implement Reinforcement Learning model for better results. Updates coming soon.

  
![Alt text](images/gameplay.gif "Gameplay")


## Update
Implemented RL Deep Q method to train the agent. The results are great, agent avoids asteroids at high game speed better than human could. However, it's not perfect and sometimes a miss happens. 


![Alt text](images/gameplay_rl.gif "Gameplay_RL")


## Instructions

To train the agent run:
`python3 train.py`

And to play the game with the trained agent run:
`python3 play.py`

To play the game manually: 
`python3 game.py`

## Requirements
python 3, pygame, pygame-menu, pytorch, numpy


