#the ai is currently set to showcase the best agent from a pre-trained generation. to train a new ai, set gensize, safesize, keepsize and showsize to higher values (e.g. 100, 10, 10, 10)
# , remove the code replacing the base agent with the pre-trained model, (you can ctrl+f for "pretrained" to find which line to do this) and run the program for multiple generations until satisfied.
# controls are:
# w - accelerate
# s - brake
# a - steer left
# d - steer right
# x - move backward to get unstuck
# r - respawn at start at the start of next generation
# t - restart ai for showcase (also allows respawn at next generation) note the ai can only drive counter-clockwise on the current tracks, so you may need to respawn multiple times to get it going the right way.

# when training the ai, use q and e to decrease and increase the maximum generation age respectively.
# note, the program already increases the max age by 1000 every 10 generations, so these controls are only needed if you want to speed up or slow down the age increase.

# when showcasing the ai, use y to set the generation age to unlimited, allowing the ai to drive indefinitely without restarting.



import pygame
import math
import numpy as np
import random
import copy
from time import sleep

import sounddevice as sd

#-----Sound-----

gap_player = 0.001
gap_AI = 0.001

min_gap=0.005
max_gap=0.05
duration = 0.01
sr = 44100

throttle_player=0
throttle_AI=0

def lerp(a: float, b: float, t: float) -> float:
    return (1 - t) * a + t * b

def sine_tone(frequency=440, duration=1.0, amplitude=0.5,offset=0, sample_rate=44100):
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, False)
    return (np.sin((2*np.pi*frequency*t+offset)) * amplitude).astype(np.float32)

def generate_playback(gap,frec, duration, sr):
    tone = sine_tone(frec,duration, 2,0, sr)
    silence = np.zeros(int(sr * gap), dtype=np.float32)
    single = np.concatenate([tone, silence])
    return single.reshape(-1, 1)

# Playback positions
player_pos = 0
ai_pos = 0

# Buffers
playback_player = np.zeros(1)
playback_AI = np.zeros(1)

def audio_callback(outdata, frames, time, status):
    global player_pos, ai_pos, playback_player, playback_AI
    for i in range(frames):
        # Player
        if player_pos >= len(playback_player):
            player_pos = 0
        sample_player = playback_player[player_pos]
        player_pos += 1

        # AI
        if ai_pos >= len(playback_AI):
            ai_pos = 0
        sample_ai = playback_AI[ai_pos]
        ai_pos += 1

        # Mix both
        outdata[i, 0] = (sample_player + sample_ai) * 0.2  # reduce volume

#-----Car and AI-----

debug_movement = False
debug_speed = 0.5
debug_turn_rate = 1

epsilon = np.exp(-10)
pygame.init()

S_Width=1080

S_Height=1080

screen = pygame.display.set_mode((0,0),pygame.RESIZABLE)
w, h = pygame.display.get_surface().get_size()
pygame.display.set_caption("Formula 1 Racing")

car = pygame.image.load("Formula_1_car.png")
car=pygame.transform.scale_by(car,0.5)

track_id = 1
track_info = [
    [ "Track.webp"
    , 1
    , [[330,585],[135,470],[188,272],[528,131],[861,178],[913,261],[791,360],[799,504],[1061,431],[1141,462],[995,582],[500,590]]
    , [80, 20, 330, 270, 240, 170, 140, 240, 270, 190, 100, 90]
    , [50,50,50,50,50,50,50,50,50,50,50,50]
    , [0,0,0,0,0,0,0,0,0,0,0,0]
    ],
    [ "Track_3.0.png"
    , 0.67
    , [[136,443],[140,144],[218,72],[367,175],[1067,65],[1215,171],[1215,584],[1140,645],[1020,605.00000], [953, 461], [854, 337], [509, 299], [448, 389], [505, 481], [692, 440], [779, 535], [723, 632], [209, 641], [149, 581]]
    , [0,337,270,248,259,212,165, 98, 51.00000, 27, 57, 122, 181, 256, 265, 179, 126, 73, 25]
    , [75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75]
    , [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    ]
]


rewards_pos = np.array(track_info[track_id][2], dtype=float)
spawn_angles = track_info[track_id][3]
rewards_radius = track_info[track_id][4]
rewards_size = track_info[track_id][5]

track=pygame.image.load(track_info[track_id][0])
track_scaled=pygame.transform.scale_by(track, track_info[track_id][1])
track_rect = track_scaled.get_rect(topleft=(0,0))



alphab = 0.2
db = 1
alphaw = 0.2
dw = 1
alphaac = 0.8
alpharc = 0.02
alphaan = 0.5
gensize = 1 #number of agents per generation set to 1 to show the best agent only
safesize = 1 #number of agents saved from each generation without mutation
keepsize = 1 #number of agents that can be chosen as parents
showsize = 1 #number of agents shown on screen. if less than or equal to safesize, will show only the best from the previous generation



maxgenage = 1000
gentimechange = 1000
genage = 10000000



distance_reward = 1

pos=rewards_pos[0].copy()
vel=np.array([0,0], dtype=float)
angle = 180+spawn_angles[0]
angle_vel=0
wheel_angle=0

acc_speed=0.0005 #strength of motor
brake_speed=0.001 #strength of brakes
max_speed=1 #max speed
steering_angle=20 #front wheel angle while steering
car_moment_inertia=1 #how hard it is for the car to start rotating. should be (much) less than 121. higher means easier to spin out. set to 1 to disable spinning out.
car_len=11 #do not change. should be 11
wheel_dynamic_friction=0.001 #grip while sliding. higher means faster recovery from drifting
wheel_static_friction=0.001 #grip while in control. higher means tighter turns without drifting



track_color = (161, 161, 161)  # Light gray track color
goal_color_1 = (0, 0, 0)
goal_color_2 = (255, 255, 255)
ray_angles = [-90, -60, -30, -15, 0, 15, 30, 60, 90]




border_bounce_multiplier=0.002


deg_to_rad=math.pi/180






def getForwardVector(angle):
    return np.array([-math.sin(angle*deg_to_rad),-math.cos(angle*deg_to_rad)], dtype=float)
def getRightVector(angle):
    return np.array([math.cos(angle*deg_to_rad),-math.sin(angle*deg_to_rad)], dtype=float)


def grip(vel, angle, angle_vel, wheel_angle): #calculates the grip force and torque on the car
    #back_forward = getForwardVector(angle)
    back_right = getRightVector(angle)

    #back_forward_comp = np.dot(vel, back_forward)
    back_right_comp = np.dot(vel, back_right)+angle_vel*deg_to_rad*car_len
    #front_forward = getForwardVector(angle+wheel_angle)
    front_right = getRightVector(angle+wheel_angle)

    #front_forward_comp = np.dot(vel-angle_vel*deg_to_rad*car_len*back_right, front_forward)
    front_right_comp = np.dot(vel-angle_vel*deg_to_rad*car_len*back_right, front_right)
    
    bb = 1 + car_len*car_len/car_moment_inertia
    bf = np.dot(back_right, front_right) - np.dot(back_right, front_right)*car_len*car_len/car_moment_inertia
    fb = np.dot(back_right, front_right) - np.dot(back_right, front_right)*car_len*car_len/car_moment_inertia
    ff = 1 + np.dot(back_right, front_right)*np.dot(back_right, front_right)*car_len*car_len/car_moment_inertia


    #x=f*b/a*e-c/a
    #y=-f/e

    #x=-c/a
    #y=-f/e


    #x*a+y*b+c=0
    #x*d+y*e+f=0
    #x=(-f+c*e/b)/(d-a*e/b)
    #y=(-f+c*d/a)/(e-b*d/a)
    #bb  fb  back_right_component
    #bf  ff  front_right_component
    if bf == 0 and fb == 0:
        des_b=-back_right_comp/bb
        des_f=-front_right_comp/ff
    else:
        des_b = (-front_right_comp+back_right_comp*ff/fb)/(bf-bb*ff/fb)
        des_f = (-front_right_comp+back_right_comp*bf/bb)/(ff-fb*bf/bb)
    if abs(des_b) <= wheel_static_friction or des_b == 0:
        back_force = des_b
    else:
        back_force = wheel_dynamic_friction*des_b/abs(des_b)
    if abs(des_f) <= wheel_static_friction or des_f == 0:
        front_force = des_f
    else:
        front_force = wheel_dynamic_friction*des_f/abs(des_f)
    if front_force+back_force < -50:
        print("bang")
        pygame.quit()
    
    return [back_right*back_force+front_right*front_force, back_force*car_len/car_moment_inertia - front_force*np.dot(back_right, front_right)*car_len/car_moment_inertia]

def raycast_distance(pos,
                     angle,
                     track_surface,
                     track_color=track_color,
                     step=10,
                     max_distance=500):
    #return 0
    x, y = pos
    rad = math.radians(angle)
    dx = -math.sin(rad) * step
    dy = -math.cos(rad) * step
    distance = 0

    while distance < max_distance:
        x += dx
        y += dy
        distance += step
        if x < 0 or y < 0 or x >= track_surface.get_width(
        ) or y >= track_surface.get_height():
            break
        if not(track_surface.get_at((int(x), int(y)))[:3] in [track_color, goal_color_1, goal_color_2]):
            break
    return distance

def check_track_collision(pos, track_surface, track_color=track_color):
    #return False
    x, y = int(pos[0]), int(pos[1])
    if x < 0 or y < 0 or x >= track_surface.get_width(
    ) or y >= track_surface.get_height():
        return True
    return not(track_surface.get_at((x, y))[:3] in [track_color, goal_color_1, goal_color_2])

def hiddenact(x): #activation function for hidden neurons (softplus)
    if x > 0:
        return np.log(1+np.exp(-x))+x
    elif x < 0:
        return np.log(1+np.exp(x))
    return np.log(2)
def outputact(x): #activation function for output neurons (scaled sigmoid)
    if x > 0:
        return 2*1/(1+np.exp(-x))-1
    elif x < 0:
        return 2*np.exp(x)/(1+np.exp(x))-1
    return 0


class agent: #neural network class
    def __init__(self, inn, outn, totn = 0, w = 0, b = 0, d = 0, i = 0, s = 0, e = 0, u = 0, a = 0, p = 0):
        self.inn = inn
        self.outn = outn
        self.totn = totn
        if totn == 0:
            self.totn = inn+outn
        self.w = w
        if w == 0:
            self.w = [[] for i in range(self.totn)] #weights
        self.b = b
        if b == 0:
            self.b = [0 for i in range(self.totn)] #biases
        self.a = a
        if a == 0:
            self.a = [0 for i in range(self.totn)] #neuron activation
        self.p = p
        if p == 0:
            self.p = [False for i in range(self.totn)] #neuron processed?
        self.d = d
        if d == 0:
            self.d = [[] for i in range(self.totn)] #dependent neurons
        self.i = i
        if i == 0:
            self.i = list(range(self.totn)) #initialized nodes
        self.s = s
        if s == 0:
            self.s = list(range(self.inn)) #nodes that a connection may start from
        self.e = e
        if e == 0:
            self.e = list(range(self.inn, self.totn)) #nodes that a connection may end at
        self.u = u
        if u == 0:
            self.u = [] #unused nodes


    def __deepcopy__(self, memo): #cursed. do not touch.
        id_self = id(self)
        _copy = memo.get(id_self)
        #if _copy is None: #(index next line)
        _copy = type(self)(copy.deepcopy(self.inn, memo), copy.deepcopy(self.outn, memo), copy.deepcopy(self.totn, memo), copy.deepcopy(self.w, memo), copy.deepcopy(self.b, memo), copy.deepcopy(self.d, memo), copy.deepcopy(self.i, memo), copy.deepcopy(self.s, memo), copy.deepcopy(self.e, memo), copy.deepcopy(self.u, memo), copy.deepcopy(self.a, memo), copy.deepcopy(self.p, memo))
        #memo[id_self] = _copy
        return _copy

    def printnet(self):
        ret = str(self.inn) + ", " + str(self.outn) + ", " + str(self.totn) + ", " + str(self.w) + ", " + str(self.b) + ", " + str(self.d) + ", " + str(self.i) + ", " + str(self.s) + ", " + str(self.e) + ", " + str(self.u) + ", " + str(self.a) + ", " + str(self.p)
        print(ret)

    def node(self, x): #processes neuron if dependent neurons are processed, activates connected neurons recursively
        if self.p[x]:
            return
        for i in self.d[x]:
            if not self.p[i]:
                return
        self.p[x] = True
        self.a[x] += self.b[x]
        if x >= self.inn+self.outn:
            self.a[x] = hiddenact(self.a[x])
        elif x >= self.inn:
            self.a[x] = outputact(self.a[x])
        for i in self.w[x]:
            self.a[i[0]] += self.a[x]*i[1]
        for i in self.w[x]:
            self.node(i[0])


    def model(self, inv): #processes the nnet
        #print(self.totn)
        self.a = [0 for i in range(self.totn)]
        self.p = [False for i in range(self.totn)]
        for i in range(self.inn):
            self.a[i] = inv[i]
        for i in self.i:
            self.node(i)
        ans = []
        for i in range(self.outn):
            ans.append(self.a[i+self.inn])
        return ans
    
    def purge(self, x): #removes all connections to and from node x
        if x < self.inn + self.outn:
            self.i.append(x)
            return
        for i in self.w[x]:
            self.d[i[0]].remove(x)
            self.b[i[0]] += i[1]*hiddenact(self.b[x])
            if self.d[i[0]] == []:
                self.purge(i[0])
        for i in self.d[x]:
            for j in range(len(self.w[i])-1, -1, -1):
                if self.w[i][j][0] == x:
                    self.w[i].pop(j)
            if self.w[i] == [] and i >= self.inn + self.outn:
                self.purge(i)
        self.s.remove(x)
        self.e.remove(x)
        self.u.append(x)
        

    def mutate(self): #self explanatory
        for i in range(self.inn, self.totn): #biases
            if np.random.rand() < alphab:
                self.b[i] += np.random.rand()*2*db-db
        for i in self.s: #weights and removing connections
            for j in range(len(self.w[i])-1, -1, -1):
                if np.random.rand() < alphaw:
                    self.w[i][j][1] += np.random.rand()*2*dw-dw
                if np.random.rand() < alpharc:
                    self.d[self.w[i][j][0]].remove(i)
                    if self.d[self.w[i][j][0]] == []:
                        self.purge(self.w[i][j][0])
                    self.w[i].pop(j)
            if self.w[i] == [] and i >= self.inn + self.outn:
                self.purge(i)
        if np.random.rand() < alphaac and self.totn-self.outn-1 >= 0: #add connections
            a = self.s[random.randint(0, len(self.s)-1)]
            b = self.e[random.randint(0, len(self.e)-1)]
            c = np.random.rand()*2*dw-dw
            if not (a in self.d[b] or a == b):
                self.w[a].append([b, c])
                if self.d[b] == []:
                    self.i.remove(b)
                self.d[b].append(a)
                self.model([0 for i in range(self.inn)])
                for i in range(self.inn, self.inn + self.outn):
                    if not self.p[i]:
                        self.d[b].remove(a)
                        if self.d[b] == []:
                            self.purge(b)
                        self.w[a].remove([b, c])
                        break
        if np.random.rand() < alphaan and self.totn-self.outn-1 >= 0: #add nodes
            a = self.s[random.randint(0, len(self.s)-1)]
            b = self.e[random.randint(0, len(self.e)-1)]
            if a != b:
                c1 = np.random.rand()*2*dw-dw
                s1 = np.random.rand()*2*db-db
                c2 = np.random.rand()*2*dw-dw
                ind = 0
                if self.u != []:
                    ind = self.u.pop()
                    self.s.append(ind)
                    self.e.append(ind)
                    self.b[ind] = s1
                    self.w[ind] = []
                    self.d[ind] = []
                else:
                    ind = self.totn
                    self.s.append(ind)
                    self.e.append(ind)
                    self.totn += 1
                    self.b.append(s1)
                    self.w.append([])
                    self.d.append([])
                self.w[a].append([ind, c1])
                self.d[ind].append(a)
                self.w[ind].append([b, c2])
                if self.d[b] == []:
                    self.i.remove(b)
                self.d[b].append(ind)
                self.model([0 for i in range(self.inn)])
                for i in range(self.inn, self.inn + self.outn):
                    if not self.p[i]:
                        self.d[b].remove(ind)
                        if self.d[b] == []:
                            self.purge(b)
                        self.w[a].remove([ind, c1])
                        self.w[ind] = []
                        self.d[ind] = []
                        self.s.remove(ind)
                        self.e.remove(ind)
                        self.u.append(ind)
                        break





class bil: #car class
    def __init__(self, net, id, reward_step):
        self.net=net
        self.pos=copy.deepcopy(rewards_pos[id])
        self.vel=np.array([0.,0.], dtype=float)
        self.angle=spawn_angles[id]
        self.angle += 5*(2*np.random.rand()-1)
        if reward_step == -1:
            self.angle += 180
        self.angle_vel=0
        self.reward_index=(id+reward_step)%len(rewards_size)
        self.score=0
        self.reward_step=reward_step
        self.collision_step = 0

    def step(self): #calculates ai decisions and car physics
        forward = getForwardVector(self.angle)
        right = getRightVector(self.angle)
        speed=np.linalg.norm(self.vel)


        

        forward_vel = np.dot(self.vel,forward)
        right_vel = np.dot(self.vel,right)


        reward_forward_distance = np.dot((rewards_pos[self.reward_index]-self.pos),forward)
        reward_right_distance = np.dot((rewards_pos[self.reward_index]-self.pos),right)
        reward_distance = np.sqrt(np.pow(reward_forward_distance,2)+np.pow(reward_right_distance,2))
        reward_forward_direction = reward_forward_distance/reward_distance
        reward_right_direction = reward_right_distance/reward_distance
        next_reward_forward_distance = np.dot((rewards_pos[(self.reward_index + self.reward_step)%len(rewards_size)]-self.pos),forward)
        next_reward_right_distance = np.dot((rewards_pos[(self.reward_index + self.reward_step)%len(rewards_size)]-self.pos),right)
        next_reward_distance = np.sqrt(np.pow(next_reward_forward_distance,2)+np.pow(next_reward_right_distance,2))
        next_reward_forward_direction = next_reward_forward_distance/next_reward_distance
        next_reward_right_direction = next_reward_right_distance/next_reward_distance
        nnext_reward_forward_distance = np.dot((rewards_pos[(self.reward_index + 2*self.reward_step)%len(rewards_size)]-self.pos),forward)
        nnext_reward_right_distance = np.dot((rewards_pos[(self.reward_index + 2*self.reward_step)%len(rewards_size)]-self.pos),right)
        nnext_reward_distance = np.sqrt(np.pow(nnext_reward_forward_distance,2)+np.pow(nnext_reward_right_distance,2))
        nnext_reward_forward_direction = nnext_reward_forward_distance/nnext_reward_distance
        nnext_reward_right_direction = nnext_reward_right_distance/nnext_reward_distance
        ray_distances = []
        for a in ray_angles:
            d = float(raycast_distance(self.pos, self.angle + a, track_scaled, track_color))/100
            ray_distances.append(d)
        inp = [forward_vel*2, right_vel*2, speed, forward_vel/(speed+epsilon), right_vel/(speed+epsilon), self.angle_vel, self.angle_vel/(speed+epsilon), reward_forward_distance/100, reward_right_distance/100, reward_distance/100, reward_forward_direction, reward_right_direction, next_reward_forward_distance/100, next_reward_right_distance/100, next_reward_distance/100, next_reward_forward_direction, next_reward_right_direction, nnext_reward_forward_distance/100, nnext_reward_right_distance/100, nnext_reward_distance/100, nnext_reward_forward_direction, nnext_reward_right_direction]
        act = self.net.model(inp + ray_distances)
        if np.pow(act[0],2) > 1 or np.pow(act[1],2) > 1:
            sleep(1)
            print(self.angle, "weights", self.net.w)
            print(self.angle, "dependencies", self.net.d)
            print(self.angle, "initials", self.net.i)
            print(self.angle, "biases", self.net.b)
            print(self.angle, "input", inp + ray_distances)
            print(self.angle, "activations", self.net.a)
            print(self.angle, "output", act)
            sleep(1)
            pygame.quit()
        if debug_movement:
            self.pos += forward*debug_speed*act[0]
            self.angle += debug_turn_rate*act[1]
        else:
            self.des_vel = (act[0]+1)*max_speed/2
            if abs(np.dot(forward, self.vel) - self.des_vel) < acc_speed:
                self.vel += (self.des_vel-np.dot(forward, self.vel))*forward
            else:
                self.vel += acc_speed*forward*(self.des_vel-np.dot(forward, self.vel))/abs(self.des_vel-np.dot(forward, self.vel))
            if act[2]+1 > 0:
                if abs(np.dot(forward, self.vel)) < (act[2]+1)*brake_speed/2:
                    self.vel -= np.dot(forward, self.vel)*forward
                else:
                    self.vel -= (act[2]+1)*brake_speed*forward*np.dot(forward, self.vel)/(abs(np.dot(forward, self.vel))*2)
            force = grip(self.vel,self.angle,self.angle_vel,act[1]*steering_angle)
            self.vel += force[0]
            self.angle_vel += force[1]
            self.pos += self.vel
            self.angle += self.angle_vel
        self.angle = self.angle%360

        if check_track_collision(self.pos, track_scaled, track_color):
            if debug_movement:
                self.pos -= forward*debug_speed*act[0]
            else:
                self.pos -= 2 * self.vel
                self.angle -= 2 * self.angle_vel
                self.vel = np.array([0.0, 0.0], dtype=float)
                self.angle_vel = 0
            if self.collision_step == 0:
                self.collision_step = 1

        

        if np.linalg.norm(rewards_pos[self.reward_index]-self.pos) < rewards_radius[self.reward_index]:
            #self.angle = np.random.rand()*360
            self.score += rewards_size[self.reward_index]
            self.score += np.linalg.norm(rewards_pos[self.reward_index]-rewards_pos[(self.reward_index-self.reward_step)%len(rewards_size)])
            self.reward_index += self.reward_step
            self.reward_index %= len(rewards_size)


#test_agent = agent(1, 1, 4, [[[2,1]],[],[[3,1]],[[2,1],[1,1]]], [0,0,0,0], [[],[3],[0,3],[2]], [0], [0, 2, 3], [1, 2, 3], [], [0,0,0,0], [False,False,False,False])
#print(test_agent.p)
#test_agent.model([1])
#print(test_agent.p)
#pygame.quit()
 

run=True
gn = 0
base_agent = agent(22+len(ray_angles), 3)
#base_agent = agent(31, 3, 36, [[], [], [], [], [], [], [[34, -0.41740312371898547], [33, 0.14458551674188658]], [[33, -0.27414594364525136], [34, -0.2742167028367666]], [], [], [], [], [], [[34, -0.36411875023991813]], [], [[33, -0.22811467618832015]], [], [], [], [], [], [], [], [], [], [], [], [], [], [[35, 0.4394414839676839]], [[34, 0.4064079848969031]], [], [], [], [[31, 0.4820515128479903]], [[32, 0.48830720474917366]]], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.220044316767244, -0.11709552979177285, -0.05435547797990914, 2.590049484116409, -0.08306251470272663], [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [34], [35], [15, 7, 6], [6, 13, 30, 7], [29]], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 34, 35], [31, 32, 33, 34, 35], [], [np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), 0, np.float64(0.0), np.float64(1.5467664544614226), np.float64(-0.3538552463265734), np.float64(1.5867261893597142), np.float64(0.974816237882595), np.float64(-0.22300964633939976), np.float64(3.386201168064389), np.float64(-3.297217258446791), np.float64(4.7263093423939155), np.float64(0.7164577946032727), np.float64(-0.6976304383785262), np.float64(2.8400518118551137), np.float64(-4.23733474084577), np.float64(5.101068515517117), np.float64(0.5567562566971728), np.float64(-0.8306759119106271), 0.5, 0.6, 1.1, 1.5, 2.1, 2.8, 3.9, 0.7, 0.6, np.float64(0.2604493056260573), np.float64(0.1387336365137466), np.float64(-0.3103339285865666), np.float64(3.6370075939115276), np.float64(0.8117098885190599)], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True])
base_agent = agent(31, 3, 50, [[], [[38, -0.4060881159666312]], [], [[32, 0.9840400268857674], [31, 0.1581992077195884]], [], [], [], [], [], [[34, -0.2670149921756859], [46, -0.6714002980370668]], [[36, 0.29825326107444305], [40, -0.8669939445236723], [45, 0.004948864599946701], [42, -1.4129139213923037]], [[37, -0.5631079339578828], [44, 0.7021133192019873]], [[31, 0.1674755187559296], [40, -1.7342555194282903]], [[33, 0.2781250571933258], [32, -0.9346315100314975]], [], [[48, -0.9757630984296735]], [[32, 0.415067388831837], [33, 0.9394838168259743]], [[33, -0.02906572607266611], [35, 1.157193541211994]], [[39, -0.4514122992274423], [36, -1.8346487302903014]], [[35, -0.30914247844959375]], [], [[33, 1.9978509816067438]], [[38, 3.145271087884237]], [[47, 0.6908402525259225]], [[32, -1.8094276057023233], [35, 0.931127774711815]], [[41, -0.16995149232536644], [46, -0.7159397883662555]], [[44, 0.7862601223271795]], [[32, 0.027412640471281335]], [[43, -0.926616467075249]], [[33, 1.8368968690427012]], [], [], [], [], [[33, 0.9988604554978082], [32, 1.4483218506801736]], [[33, -1.8545859405795293]], [[39, -0.8281879296722121]], [[41, -2.496665882897522]], [[35, 3.5793324312971806], [36, -0.6093235831741193]], [[35, 0.8528853747705218], [40, 0.34652795483445176]], [[32, -0.2726467559729835]], [[36, 0.11301318435289254]], [[45, 0.7053423006753528]], [[41, -0.8533352841973583]], [[31, 0.15830380608336503]], [[38, 0.06616341570171769]], [[41, -0.2439604442129406], [49, 0.9167651874530587]], [[40, 0.45931768309221255]], [[36, 0.33535099254106027]], [[44, -0.24058652146692583]]], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.6409565742983863, 0.12443604827648436, np.float64(-1.1816455084839654), 0.41808959848898386, np.float64(1.3900440596394732), np.float64(1.9742466244023582), 0.4654914129310661, np.float64(1.1404212390214299), 1.1040285252499076, 1.0139012783631387, 0.7089035558595158, 1.483268357011466, 1.2628020989546629, 0.22681853582180755, -0.6726159760737167, 1.1871171680887902, 0.6357076031907536, -0.004466054322868951, 0.6304851604047585], [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [12, 3, 44], [16, 24, 3, 34, 13, 40, 27], [17, 13, 35, 21, 16, 34, 29], [9], [19, 38, 39, 24, 17], [10, 18, 41, 38, 48], [11], [22, 1, 45], [18, 36], [10, 12, 39, 47], [25, 37, 43, 46], [10], [28], [11, 26, 49], [10, 42], [25, 9], [23], [15], [46]], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 35, 34, 37, 38, 39, 36, 41, 40, 43, 44, 45, 42, 46, 47, 48, 49], [31, 32, 33, 35, 34, 37, 38, 39, 36, 41, 40, 43, 44, 45, 42, 46, 47, 48, 49], [], [np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), 0, np.float64(0.0), np.float64(0.7756338546651955), np.float64(-0.34408156518071453), np.float64(0.8485281374238569), np.float64(0.9140932639193681), np.float64(-0.40550401336760716), np.float64(2.5401375048019457), np.float64(-5.172562368565366), np.float64(5.762612254871917), np.float64(0.44079618625293127), np.float64(-0.8976072204393585), np.float64(1.8364252755192596), np.float64(-6.043934331826746), np.float64(6.3167713271892305), np.float64(0.29072213958652393), np.float64(-0.9568075237759337), 0.8, 0.8, 1.0, 1.1, 1.3, 1.5, 2.0, 0.6, 0.6, np.float64(-0.024940899836912944), np.float64(0.9625584958020865), np.float64(-0.9999999999999896), np.float64(0.7934851048632471), np.float64(15.824544591518451), np.float64(11.241050792972938), np.float64(1.0990701807685306), np.float64(3.7234590118785373), np.float64(0.004171605550873963), np.float64(0.0293312932824182), np.float64(0.06211730249619196), np.float64(0.793601993933545), np.float64(0.4408892081546123), np.float64(1.0464282923257562), np.float64(0.6404478217769741), np.float64(0.6117010491198012), np.float64(1.454364095076609), np.float64(0.4992816959268025), np.float64(1.456581099560855)], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]) #pretrained agent. Uncomment to use, and comment to go back to random agents.
gs = [[[0, i, copy.deepcopy(base_agent)] for i in range(gensize)]]
ps = []
agents = [bil(gs[gn][i][2], random.randint(0, len(rewards_size)-1), random.randint(0,1)*2-1) for i in range(gensize)]
edown = False
qdown = False
collidednum = 0

#use only when running one pretrained AI and one playercontroled car (comment when not in use)
stream = sd.OutputStream(samplerate=sr, channels=1,callback=audio_callback,blocksize=256)
stream.start()
#-----Audio End-----

while run:
    screen.fill((0,0,0))
    screen.blit(track_scaled,track_rect)
    #car_forward_x = -math.sin(angle*math.pi/180)
    #car_forward_y = math.cos(angle*math.pi/180)

    if genage >= maxgenage: #create next generation

        genage = 0
        collidednum = 0
        for i in range(gensize):
            gs[gn][i][0] = -agents[i].score-distance_reward*(np.linalg.norm(rewards_pos[agents[i].reward_index]-rewards_pos[(agents[i].reward_index-agents[i].reward_step)%len(rewards_size)])-np.linalg.norm(rewards_pos[agents[i].reward_index]-agents[i].pos))
        gs[gn] = sorted(gs[gn])
        #print(gs[gn])
        ps.append([])
        for i in range(gensize):
            ps[gn].append(gs[gn][i][0])
        #print(ps[gn][0])
        gs.append(gs[gn])
        for i in range(safesize, gensize):
            gs[gn+1][i] = [0, gs[gn][i][1], copy.deepcopy(gs[gn][random.randint(0, keepsize-1)][2])]
            for j in range(random.randint(0, random.randint(0, 4))):
                gs[gn+1][i][2].mutate()
        gn += 1
        agents = [bil(gs[gn][i][2], random.randint(0, len(rewards_size)-1), random.randint(0,1)*2-1) for i in range(gensize)]
    for i in range(gensize):
        agents[i].step()
        if agents[i].collision_step == 1 and i < safesize*2: #ends the generation early if enough top agents collide in order to not waste time
            agents[i].collision_step = 2
            collidednum += 1
            if collidednum >= safesize*3/2:
                genage = 10*maxgenage

    for i in range(showsize):
        ai_car_rotated = pygame.transform.rotate(car, int(agents[i].angle))
        ai_car_rect=ai_car_rotated.get_rect(center=(agents[i].pos[0],agents[i].pos[1]))
        screen.blit(ai_car_rotated,ai_car_rect)

    genage+=1

    

    

    car_rotated=pygame.transform.rotate(car,int(angle))
    car_rect=car_rotated.get_rect(center=(pos[0],pos[1]))
    screen.blit(car_rotated,car_rect)
    #pygame.draw.rect(screen,(255,0,0,0),car)
    forward = getForwardVector(angle)
    right = getRightVector(angle)
    speed=np.linalg.norm(vel)

    #for a in ray_angles:
    #    pygame.draw.line(screen, (255, 0, 0), pos, pos+getForwardVector(angle+a)*raycast_distance(pos, angle+a, track_scaled, track_color), 2)
    #print(raycast_distance(pos, angle, track_scaled, track_color)/100)
    #pygame.draw.line(screen, (255, 0, 0), pos, pos+getForwardVector(angle)*raycast_distance(pos, angle, track_scaled, track_color), 2)
    key= pygame.key.get_pressed()
    wheel_angle = 0
    if key[pygame.K_a]: #turn left
        if debug_movement:
            angle += debug_turn_rate
        elif wheel_angle < steering_angle and wheel_angle >= 0:
            wheel_angle += steering_angle
    elif key[pygame.K_d]: #turn right
        if debug_movement:
            angle -= debug_turn_rate
        elif wheel_angle > -steering_angle and wheel_angle <= 0:
            wheel_angle -= steering_angle
    if key[pygame.K_e] and (not edown): #increase maxgenage
        edown = True
        maxgenage += gentimechange
    if (not key[pygame.K_e]) and edown:
        edown = False
    if key[pygame.K_q] and (not qdown): #decrease maxgenage
        qdown = True
        maxgenage -= gentimechange
    if (not key[pygame.K_q]) and qdown:
        qdown = False
    if key[pygame.K_x]: #move backward to get unstuck
        pos -= forward*0.2
        if check_track_collision(pos, track_scaled, track_color):
            pos += forward
    if genage==1:
        #print("angle:", angle, "pos:", pos)
        
        if key[pygame.K_r]: #respawn
            a = random.randint(0,0)#, len(rewards_size)-1)
            pos=copy.deepcopy(rewards_pos[a])
            vel=np.array([0.,0.], dtype=float)
            angle = 180 + spawn_angles[a]
            angle_vel=0
        if gn % 10 == 0 and maxgenage < 15000:
            maxgenage += gentimechange
        if key[pygame.K_p] or gn % 50 == 0:
            print()
            print("Generation:", gn, "Max Age:", maxgenage, "Best Score:", -gs[gn][0][0])
            print()
            agents[0].net.printnet()
            print()

    
    des_vel = 0.
    if key[pygame.K_w]: #move forward
        if debug_movement:
            pos += forward*debug_speed
        else:
            des_vel = max_speed
    if abs(np.dot(forward, vel)-des_vel) < acc_speed:
        vel += forward*(des_vel-np.dot(forward, vel))
    else:
        vel += forward*acc_speed*(des_vel-np.dot(forward,vel))/abs(des_vel-np.dot(forward,vel))
    if key[pygame.K_s]: #brake
        if debug_movement:
            pos -= forward*debug_speed
        elif np.dot(forward, vel) - brake_speed >= 0:
            vel -= brake_speed*forward
        elif np.dot(forward, vel) > 0:
            vel -= forward*np.dot(forward, vel)
    if key[pygame.K_t]: #restart ais for showcase
        genage = maxgenage
    if key[pygame.K_y]: #unlimited age for showcase
        maxgenage = np.pow(2, 31)-1
    if key[pygame.K_m]: #toggle debug movement
        debug_movement = not debug_movement
    
    if check_track_collision(pos, track_scaled, track_color):
        pos -= 2*vel
        vel = np.array([0.0, 0.0], dtype=float)

    if not debug_movement: #apply grip physics
        force = grip(vel,angle,angle_vel,wheel_angle)
        vel += force[0]
        angle_vel += force[1]
        pos += vel
        angle += angle_vel
    angle = angle%360

    #Changing the "RPM" of the motors (Changing the gap between pistons fireing)
    throttle_player = np.linalg.norm(vel)/max_speed
    throttle_AI = np.linalg.norm(agents[0].vel)/max_speed

    #throttle*2*(1-throttle*0.5) = x*2*(1-x*0.5)= x*2-x*x
    gap_player=lerp(min_gap,max_gap,1-(throttle_player*2-throttle_player**2))
    gap_player+=np.random.uniform(0.0001,0.0005)
    playback_player = generate_playback(gap_player,100, duration, sr)

    gap_AI=lerp(min_gap,max_gap,1-(throttle_AI*2-throttle_AI**2))
    gap_AI+=np.random.uniform(0.0001,0.0005)
    playback_AI = generate_playback(gap_AI,206, duration, sr)


    if key[pygame.K_ESCAPE]:
        run = False

    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            run = False
    pygame.display.update()


pygame.quit()
