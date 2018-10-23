import pygame
import time

class Drone_agent():
    def _init_(self):
        self.drone_speed
        self.x_change
        self.y_change
        self.drone_pos_x
        self.drone_pos_y
        self.thing_pos_x
        self.thing_pos_y
        self.my_drone.thing_size_x
        self.my_drone.thing_size_y

    def move_left(self):
        #self.x_change -= self.drone_speed
        self.x_change = -20

    def move_right(self):
        #self.x_change += self.drone_speed
        self.x_change = 20

    def move_up(self):
        self.y_change = -self.drone_speed

    def move_down(self):
        self.y_change = +self.drone_speed

    def move_left_big(self):
        self.x_change -= 4* self.drone_speed

    def avoid_things(self):
        #if drone is
        print ("drone pos", self.drone_pos_x)
        print ("thing pos", self.thing_pos_x)
        print ("distance",self.drone_pos_x - self.thing_pos_x)
        if (self.drone_pos_x - self.thing_pos_x) > 0 and self.drone_pos_x < 600:
        #abs(self.drone_pos_x - self.thing_pos_x) < 200:
        #and abs(self.drone_pos_x - self.thing_pos_x) > 20 and self.drone_pos_x > 400:
            self.move_right()

            #time.sleep(.100)

            #print ("drone pos", self.drone_pos_x)
            #print ("change in x", self.x_change)
            print("moving right", abs(self.drone_pos_x - self.thing_pos_x))


        elif self.drone_pos_x - self.thing_pos_x < 0 and self.drone_pos_x > 80:
        #abs(self.drone_pos_x - self.thing_pos_x) > 200:
        # and abs(self.drone_pos_x - self.thing_pos_x) > 20 and self.drone_pos_x < 400:
            self.move_left()

            #time.sleep(.100)
            #print ("drone pos", self.drone_pos_x)
            #print ("change in x", self.x_change)
            print("moving left", abs(self.drone_pos_x - self.thing_pos_x))

        else:
        # abs(self.drone_pos_x - self.thing_pos_x) > 150:
            self.x_change = 0
            #print("not moving")

    def clear_change(self):
        self.x_change = 0
        #time.sleep(.100)
        print("clearing speed")

        #elif (self.drone_pos_x + 70 > self.thing_pos_x + self.thing_size_x):
        #    self.move_right()
        #    print('moving right')
        #if (self.drone_pos_y > self.thing_pos_y + self.thing_size_y):
        #    self.move_right()
