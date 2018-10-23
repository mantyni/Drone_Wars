import pygame
import time
import random
from agent import Drone_agent
from coded_policy import Policy

pygame.init()

display_width = 800
display_height = 600

gameDisplay = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption('test')

droneImg = pygame.image.load('camera.png')
droneImg = pygame.transform.scale(droneImg, (int(display_width*0.08),int(display_height*0.1)))

my_drone = Drone_agent()
#my_policy = Policy()

black = (0,0,0)
white = (255,255,255)
red = (255,0,0)
dark_red = (150,0,0)
green = (0,255,0)
dark_green = (0,150,0)

drone_width = 70
drone_height = 70

clock = pygame.time.Clock()
crashed = False

def drone(x,y):
    gameDisplay.blit(droneImg, (x,y))

def things_dodged(count):
    font = pygame.font.SysFont(None, 25)
    text = font.render("Score: "+str(count), True, black)
    gameDisplay.blit(text,(0,0))

def things(thingx, thingy, thingw, thingh, color):
    pygame.draw.rect(gameDisplay, color, [thingx, thingy, thingw, thingh])

def text_objects(text, font):
    textSurface = font.render(text, True, black)
    return textSurface, textSurface.get_rect()

def message_display(text):
    largeText = pygame.font.Font('freesansbold.ttf',80)
    TextSurf, TextRect = text_objects(text, largeText)
    TextRect.center = ((display_width/2),(display_height/2))
    gameDisplay.blit(TextSurf, TextRect)

    pygame.display.update()

    #time.sleep(2)

    #game_loop

def crash():
    message_display('CRASH AND BURN')
    time.sleep(2)

def button(msg,x,y,w,h,ic,ac,action=None):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    #print(click)
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


def game_intro():
    intro = True
    while intro:
        for event in pygame.event.get():
            #print(event)
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        gameDisplay.fill(white)
        largeText = pygame.font.Font('freesansbold.ttf', 100)
        TextSurf, TextRect = text_objects('Drone surf', largeText)
        TextRect.center = ((display_width/2),(display_height)/2)
        gameDisplay.blit(TextSurf, TextRect)

        mouse = pygame.mouse.get_pos()

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


#def environment(dronex, droney, thingx, thingy):
#    return


def game_loop():

    drone_x = (display_width * 0.5)
    drone_y = (display_height * 0.5)

    my_drone.x_change = 0
    my_drone.y_change = 0
    my_drone.drone_speed = 5

    thing_startx = random.randrange(0, display_width)
    thing_starty = -600
    thing_speed = 30
    thing_width = 100
    thing_height = 100
    test = 0

    thingCount = 1
    dodged = 0

    gameExit = False

    while not gameExit:

        my_drone.drone_pos_x = drone_x
        my_drone.drone_pos_y = drone_y
        my_drone.thing_pos_x = thing_startx
        my_drone.thing_pos_y = thing_starty
        my_drone.thing_size_x = thing_width
        my_drone.thing_size_y = thing_height

        #my_drone.x_change = 0
        #my_drone.y_change = 0

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
                    my_drone.x_change = 0
                    my_drone.y_change = 0
                elif event.key == pygame.K_LSHIFT:
                    my_drone.drone_speed = my_drone.drone_speed / 2

        drone_x += my_drone.x_change
        drone_y += my_drone.y_change


        #my_drone.move_left_big()

        gameDisplay.fill(white)

        things(thing_startx, thing_starty, thing_width, thing_height, red)
        thing_starty += thing_speed
        drone(drone_x,drone_y)
        things_dodged(dodged)

        #thing_speed = 1.2*thing_speed

        if (drone_x > display_width - drone_width or drone_x < 0) or \
            (drone_y > display_height - drone_height or drone_y < 0):
            crash()
            gameExit = True

        if thing_starty > display_height:
            thing_starty = 0 - thing_height
            thing_startx = random.randrange(0, display_width)
            if thing_speed < 50:
                thing_speed = 1.15 * thing_speed
            dodged += 1
            #if thing_width < display_width / 4 and dodged > 1:
            #    thing_width += (dodged *1.2)

        if drone_y < thing_starty+thing_height:
            #print('y crossover')

            if drone_x > thing_startx and drone_x < thing_startx + thing_width or drone_x+drone_width > thing_startx and drone_x + drone_width < thing_startx+thing_width:
                dodged -= 1
                thing_width = 100
                #print('x crossover')
                #crash()
                #gameExit = True

        my_drone.avoid_things()

        #print("timer", pygame.time.get_ticks())
        #if pygame.time.get_ticks() - test > 300:
        #    my_drone.clear_change()
        #    test = pygame.time.get_ticks()


        pygame.display.update()
        clock.tick(20)



#game_intro()
game_loop()
pygame.quit()
quit()
