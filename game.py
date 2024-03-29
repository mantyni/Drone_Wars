# This script is redundant
# Use it only for testing the manual game play

import pygame
import time
import random
from drone import Drone
from asteroid import Asteroid
import pygame_menu
from random import randrange, choice, randint
import numpy as np
from pygame.surfarray import array3d

pygame.init()

# Initiate game clock
clock = pygame.time.Clock()

# Display parameters
display_width = 800
display_height = 600
fps = 30
flags = pygame.HIDDEN
#flags = pygame.SHOWN
gameDisplay = pygame.display.set_mode((display_width,display_height), flags) # This is redundant due to pygame_functions package. TODO: test and rmeove. 
#gameDisplay = pygame.display.set_mode((1,1)) # This is redundant due to pygame_functions package. TODO: test and rmeove. 

# Define game colours
black = (0,0,0)
white = (255,255,255)
dark_red = (150,0,0)
green = (0,255,0)
dark_green = (0,150,0)
red = (255,0,0)

ai_mode = True

# Initialise display using pygame_functions
#screenSize(800,600)
#setAutoUpdate(False)
#setBackgroundImage(["images/bg2.jpg", "images/bg2.jpg"])

# Pygame menu
#surface = pygame.display.set_mode((display_width, display_height))

pygame.display.set_caption('Drone Wars')

# Game functions
def scoreboard(count):

    font = pygame.font.SysFont(None, 25)
    text = font.render("Score: "+str(count), True, black)
    gameDisplay.blit(text,(0,0))


def text_objects(text, font):

    textSurface = font.render(text, True, black)
    return textSurface, textSurface.get_rect()


def message_display(text, size):

    largeText = pygame.font.Font('freesansbold.ttf',size)
    TextSurf, TextRect = text_objects(text, largeText)
    TextRect.center = ((display_width/2),(display_height/2))
    gameDisplay.blit(TextSurf, TextRect)
    #pygame.display.update()


def crash():

    message_display('Game Over', 80)
    pygame.display.update()
    time.sleep(2)


def button(msg,x,y,w,h,ic,ac,action=None):

    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()

    if x+w > mouse[0] > x and y+h > mouse[1] > y:
        pygame.draw.rect(gameDisplay, ac,(x,y,w,h))

        if click[0] == 1 and action != None:
            action()
    else:
        pygame.draw.rect(gameDisplay, ic,(x,y,w,h))

    smallText = pygame.font.SysFont("comicsansms",20)
    textSurf, textRect = text_objects(msg, smallText)
    textRect.center = ( (x+(w/2)), (y+(h/2)) )
    gameDisplay.blit(textSurf, textRect)


def avoid_obstacles(drone, obstacle):
   
    if (drone.x - obstacle.x) > 0 and drone.x < 600:
        drone.move_right()

    elif drone.x - obstacle.x < 0 and drone.x > 80:
        drone.move_left()

    else:
        drone.x_change = 0


def out_of_bounds(drone, display_width, display_height):

    if (drone.x > display_width - drone.drone_width or drone.x < 0) or \
        (drone.y > display_height - drone.drone_height or drone.y < 0):
        
        return True 


def collision(drone, obstacle):

        if (drone.y < obstacle.y + obstacle.height):

            if (drone.x > obstacle.x
                 and drone.x < obstacle.x + obstacle.width or drone.x + drone.drone_width > obstacle.x 
                 and drone.x + drone.drone_width < obstacle.x + obstacle.width):
                
                return True    


def set_ai(value, value2):
    global ai_mode

    if value2 == 1:
        ai_mode = True
        print("AI mode is: ", ai_mode)
    if value2 == 0:
        ai_mode = False
        print("AI mode is: ", ai_mode)


# Game menu function is not used anymore since it's superseeded by pygame-menu
def game_menu():

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        gameDisplay.fill(white)
        message_display("Drone Wars", 100)
        mouse = pygame.mouse.get_pos()

        # Mouse hover over buttons
        if 150+100 > mouse[0] > 150 and 450+50 > mouse[1] > 450:
            pygame.draw.rect(gameDisplay, dark_green,(150,450,100,50))
        else:
            pygame.draw.rect(gameDisplay, green,(150,450,100,50))

        if 550+100 > mouse[0] > 450 and 450+50 > mouse[1] > 450:
            pygame.draw.rect(gameDisplay, dark_red,(550,450,100,50))
        else:
            pygame.draw.rect(gameDisplay, red,(550,450,100,50))

        button("START",150,450,100,50,green,dark_green,game_loop)
        button("QUIT",550,450,100,50,red,dark_red,quit)

        pygame.display.update()
        clock.tick(15)


def game_loop():

    # Initialise drone and obstacle objects
    my_drone = Drone(gameDisplay)
    my_obstacle = Obstacle(gameDisplay)

    my_drone.x = display_width * 0.5
    my_drone.y = display_height * 0.5    
    my_obstacle.x = random.randrange(0, display_width)
    my_obstacle.y = -600
    
    score = 0
    gameExit = False
    movingLeft = False
    movingRight = False
    movingUp = False
    movingDown = False

    while not gameExit:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    movingLeft = True
                elif event.key == pygame.K_RIGHT:
                    movingRight = True
                elif event.key == pygame.K_UP:
                    movingUp = True
                elif event.key == pygame.K_DOWN:
                    movingDown = True
                elif event.key == pygame.K_LSHIFT:
                    my_drone.drone_speed = my_drone.drone_speed * 2
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT \
                    or event.key == pygame.K_DOWN or event.key == pygame.K_UP:
                    movingLeft = False
                    movingRight = False
                    movingDown = False
                    movingUp= False
                elif event.key == pygame.K_LSHIFT:
                    my_drone.drone_speed = my_drone.drone_speed / 2

        # Moving the drone based on arrow keys pressed 
        if movingLeft:
            my_drone.move_left()
        if movingRight:
            my_drone.move_right()
        if movingUp:
            my_drone.move_up()
        if movingDown:
            my_drone.move_down()

        # Update drone position 
        my_drone.update()

        # Update obstacle position. Move obstacle down the screen.
        my_obstacle.update()

        # Detect if drone left the display bounds, then game over
        if out_of_bounds(my_drone, display_width, display_height):
            crash()
            gameExit = True

        # Detect if obstacle went to the bottom of the screen, then reset y & x coordinates to start from the top again at a random x coordinate. 
        # Increase obstacles speed as the game progresses. 
        if my_obstacle.y > display_height:
            my_obstacle.reset()
            #if my_obstacle.speed < 50:
            #    my_obstacle.speed = 1.15 * my_obstacle.speed
            score += 1

        # Detect when obstacle collides with the drone and reduce the score 
        if collision(my_drone, my_obstacle):
            score -= 1 

        # AI to avoid obstacles. 
        if ai_mode:
            avoid_obstacles(my_drone, my_obstacle)

        # Move the background. 
        ##scrollBackground(0, 5)

        # Draw white background and Quit button
        gameDisplay.fill(white) # Comment this out if using scrolBackground
        button("QUIT",650,500,100,50,red,dark_red,quit)

        # Draw obstacles
        my_obstacle.draw()

        # Draw_drone
        my_drone.draw()

        # Draw score
        scoreboard(score)

        pygame.display.update()
        clock.tick(fps) 
        

# Define PyGame menu components
menu = pygame_menu.Menu('Drone Wars', 800, 600, theme=pygame_menu.themes.THEME_DARK)
menu.add.selector('Scripted AI :', [('On', 1), ('Off', 0)], onchange=set_ai)
menu.add.button('Play', game_loop)
menu.add.button('Quit', pygame_menu.events.EXIT)


if __name__ == "__main__":

    # Uncomment to play manually or use scripted AI
    surface = pygame.display.set_mode((display_width, display_height))
    menu.mainloop(surface)