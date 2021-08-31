from drone import Drone

class Policy(Drone):
    def _init_(self, drone_pos_x, drone_pos_y, obstacle_pos_x, obstacle_pos_y):
        self.obstacle_pos_x = obstacle_pos_x
        self.obstacle_pos_y = obstacle_pos_y
        #super().__init__(drone_pos_x, drone_pos_y)
        #super(Drone, self).__init__()
        Drone.__init__(self, drone_pos_x, drone_pos_y)
        self.drone_pos_x = drone_pos_x
        self.drone_pos_y = drone_pos_y


    def avoid_obstacles(self):
        #Debug:
        print ("drone pos", self.drone_pos_x)
        print ("thing pos", self.obstacle_pos_x)
        print ("distance", self.drone_pos_x - self.obstacle_pos_x)
        if (self.drone_pos_x - self.obstacle_pos_x) > 0 and self.drone_pos_x < 600:
        #abs(self.drone_pos_x - self.obstacle_pos_x) < 200:
        #and abs(self.drone_pos_x - self.obstacle_pos_x) > 20 and self.drone_pos_x > 400:
            self.move_right()
            #time.sleep(.100)
            #print ("drone pos", self.drone_pos_x)
            #print ("change in x", self.x_change)
            print("moving right", abs(self.drone_pos_x - self.obstacle_pos_x))


        elif self.drone_pos_x - self.obstacle_pos_x < 0 and self.drone_pos_x > 80:
        #abs(self.drone_pos_x - self.obstacle_pos_x) > 200:
        # and abs(self.drone_pos_x - self.obstacle_pos_x) > 20 and self.drone_pos_x < 400:
            self.move_left()

            #time.sleep(.100)
            #print ("drone pos", self.drone_pos_x)
            #print ("change in x", self.x_change)
            print("moving left", abs(self.drone_pos_x - self.obstacle_pos_x))

        else:
        # abs(self.drone_pos_x - self.obstacle_pos_x) > 150:
            self.x_change = 0
            #print("not moving")
