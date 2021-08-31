#!/usr/bin/env python3

import pygame
import time
import random
from drone import Drone
from obstacle import Obstacle

#initialise game window
pygame.init()

# Initiate game clock
clock = pygame.time.Clock()

# Display parameters
display_width = 800
display_height = 600
fps = 20
gameDisplay = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption('Drone Surf')

# Define game colours
black = (0,0,0)
white = (255,255,255)
dark_red = (150,0,0)
green = (0,255,0)
dark_green = (0,150,0)
red = (255,0,0)


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

    #Debug:
    print ("drone x pos: ", drone.x)
    print ("obstacle x pos: ", obstacle.x)
    print ("distance", drone.x - obstacle.x)
    
    if (drone.x - obstacle.x) > 0 and drone.x < 600:
        drone.move_right()
        print("moving right", abs(drone.x - obstacle.x))

    elif drone.x - obstacle.x < 0 and drone.x > 80:
        drone.move_left()
        print("moving left", abs(drone.x - obstacle.x))

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


def game_menu():
    intro = True
    while intro:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        gameDisplay.fill(white)

        message_display("Drone Surf", 100)
        #largeText = pygame.font.Font('freesansbold.ttf', 100)
        #TextSurf, TextRect = text_objects('Drone surf', largeText)
        #TextRect.center = ((display_width/2),(display_height)/2)
        #gameDisplay.blit(TextSurf, TextRect)

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


    while not gameExit:
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    my_drone.move_left()
                elif event.key == pygame.K_RIGHT:
                    my_drone.move_right()
                elif event.key == pygame.K_UP:
                    my_drone.move_up()
                elif event.key == pygame.K_DOWN:
                    my_drone.move_down()
                elif event.key == pygame.K_LSHIFT:
                    my_drone.drone_speed = my_drone.drone_speed * 2
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT \
                    or event.key == pygame.K_DOWN or event.key == pygame.K_UP:
                    my_drone.x_change = 0 # Reset movement 
                    my_drone.y_change = 0
                elif event.key == pygame.K_LSHIFT:
                    my_drone.drone_speed = my_drone.drone_speed / 2


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
            if my_obstacle.speed < 50:
                my_obstacle.speed = 1.15 * my_obstacle.speed
            score += 1

        # Detect when obstacle collides with the drone and reduce the score 
        if collision(my_drone, my_obstacle):
            score -= 1 

        # AI to avoid obstacles
        avoid_obstacles(my_drone, my_obstacle)

        # Draw white background and Quit button
        gameDisplay.fill(white)
        button("QUIT",650,500,100,50,red,dark_red,quit)

        # Draw an obstacle, drone, score
        my_obstacle.draw()

        # Draw_drone(my_drone)
        my_drone.draw()

        # Draw score
        scoreboard(score)

        pygame.display.update()
        clock.tick(fps) 


if __name__ == '__main__':
    game_menu()
    #game_loop() # Skip main menu 
    #TODO: send to the main menu instead of quiting the game when in game loop