import pygame
from pygame.locals import *
import numpy.random as random
import os
import numpy as np

# Set the window position to the top-left corner
# os.environ['SDL_VIDEO_WINDOW_POS'] = '0,0'


# Constants define the colour and size of game window and paddle's parameter
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
LIGHT_BLUE = (173, 20, 230)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
LIGHT_BLUE2 = (173, 216, 230)

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
PADDLE_WIDTH = 15
PADDLE_HEIGHT = 60
PADDLE_SPEED = 15
BALL_WIDTH = 15
BALL_HEIGHT = 15

# States
START = 0
PLAYING = 1
PAUSED = 2
GAME_OVER = 3

MAX_SCORE = 3

LEFT = -1
RIGHT = 1


BALL_INITIAL_SPEED = 5


LIMIT_SPEED = False


Y_RANDOM_RANGE = (0, 2) # the range of randomisation when a collision happens between the ball and the paddle


APPROACH_REWARD = 1
CORRECT_BALL_REWARD = 100
INCORRECT_BALL_REWARD = 50
DISTANCE_MAX_REWARD = 0.5
DISTANCE_REWARD_PUNISHMENT_RATIO = 1.5

# INCORRECT_BALL_PUNISHMENT = 20
# MISS_CORRECT_BALL_PUNISHMENT = 100
# MISS_INCORRECT_BALL_PUNISHMENT = 50
CORNER_PUNISHMENT = 0.5
STAY_PUNISHMENT = 0.5
OVERLAP_PUNISHMENT = 0.5

STAY_PUNISHMENT_THREASHOLD = 0.2
OVERLAP_PUNISHMENT_INTERVAL = 60


#show the game window

# RENDER = False
RENDER = True
HUMAN_CONTROL = False

if not RENDER:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# Initialize pygame
pygame.init()

def generate_center_position():
    x = SCREEN_WIDTH // 2 - BALL_WIDTH // 2
    y = SCREEN_HEIGHT // 2 - BALL_HEIGHT // 2
    return x, y

def get_reward_from_distance(distance: float, correct: bool, threshold: float = SCREEN_HEIGHT/4, max_reward: float = DISTANCE_MAX_REWARD, rp_ratio: float = DISTANCE_REWARD_PUNISHMENT_RATIO):
    a = max_reward/(threshold**2)
    
    if distance < threshold:
        result = rp_ratio * a * (distance-threshold)**2
    else: 
        result = -a / rp_ratio * (distance - threshold)**2
    
    if correct:
        return 2 * result
    else:
        return result 
    
def get_punishment_from_staying(total_action: int, N: int) -> float:
    return STAY_PUNISHMENT if total_action / N < STAY_PUNISHMENT_THREASHOLD else 0


class PongGame:
    def __init__(self):
        self.current_server = LEFT
        
        self.done = False
        self.reward_right_1 = 0
        self.reward_right_2 = 0
        self.reward_left_1 = 0
        self.reward_left_2 = 0

        self.left_paddle_1_total_action = 0
        self.left_paddle_2_total_action = 0
        self.right_paddle_1_total_action = 0
        self.right_paddle_2_total_action = 0

        # counting for each paddel collide with a ball
        self.right_paddle1_count = 0
        self.right_paddle2_count = 0
        self.left_paddle1_count = 0
        self.left_paddle2_count = 0

        # number of each ball out of range
        self.ball1_count  = 0
        self.ball2_count  = 0

        # running time for the game
        self.time = 0

        self.state = START
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Pong State Machine with Centered Balls')

        # Balls with independent velocities
        self.ball1 = pygame.Rect(*generate_center_position(), BALL_WIDTH, BALL_HEIGHT)
        # self.ball1_speed_x = random.choice([-3, -2, -1, 1, 2, 3])
        # self.ball1_speed_y = random.choice([-3, -2, -1, 1, 2, 3])
        # self.ball1_speed_x = 3
        # self.ball1_speed_y = 3


        self.ball2 = pygame.Rect(*generate_center_position(), BALL_WIDTH, BALL_HEIGHT)
        # self.ball2_speed_x = random.choice([-3, -2, -1, 1, 2, 3])
        # self.ball2_speed_y = random.choice([-3, -2, -1, 1, 2, 3])
        # self.ball2_speed_x = -3
        # self.ball2_speed_y = -3

        (self.ball1_speed_x, self.ball1_speed_y), (self.ball2_speed_x, self.ball2_speed_y) = [self.generate_velocity(direction=i) for i in [-1, 1]]

        # Paddles' position and width and length
        self.left_paddle1 = pygame.Rect(5, SCREEN_HEIGHT // 4 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.left_paddle2 = pygame.Rect(5, 3 * SCREEN_HEIGHT // 4 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.right_paddle1 = pygame.Rect(SCREEN_WIDTH - 5 - PADDLE_WIDTH, SCREEN_HEIGHT // 4 - PADDLE_HEIGHT // 2,
                                         PADDLE_WIDTH, PADDLE_HEIGHT)
        self.right_paddle2 = pygame.Rect(SCREEN_WIDTH - 5 - PADDLE_WIDTH, 3 * SCREEN_HEIGHT // 4 - PADDLE_HEIGHT // 2,
                                         PADDLE_WIDTH, PADDLE_HEIGHT)

        # Scores
        self.left_score = 0
        self.right_score = 0
        self.font = pygame.font.SysFont(None, 55)

        self.last_left_paddle1_y = self.left_paddle1.centery
        self.last_left_paddle2_y = self.left_paddle2.centery
        self.last_right_paddle1_y = self.right_paddle1.centery
        self.last_right_paddle2_y = self.right_paddle2.centery

        self.left_paddle1_last_moved = 0
        self.left_paddle2_last_moved = 0
        self.right_paddle1_last_moved = 0
        self.right_paddle2_last_moved = 0


        self.left_paddle_last_nonoverlap = 0
        self.right_paddle_last_nonoverlap = 0
        
        # punishments
        self.left_paddle1_stay = 0
        self.left_paddle1_corner = 0
        
        self.left_paddle2_stay = 0
        self.left_paddle2_corner = 0
        
        self.right_paddle1_stay = 0
        self.right_paddle1_corner = 0
        
        self.right_paddle2_stay = 0
        self.right_paddle2_corner = 0
        
        self.left_overlap = 0
        self.right_overlap = 0

    def generate_velocity(self, speed=BALL_INITIAL_SPEED, num=2, direction=None):
        if direction is None:
            direction = random.choice((-1, 1))
        magnitude = speed * direction
        angle = random.uniform(-np.pi / 4, np.pi / 4)
        vel_x = magnitude * np.cos(angle)
        vel_y = magnitude * np.sin(angle)
        return vel_x, vel_y

    # def get_current_server_pos(self):
    #     if self.current_server == LEFT:
    #         return (self.left_paddle1.center, self.left_paddle2.center)
    #     else:
    #         return (self.right_paddle1.center, self.right_paddle2.center)

    def reset(self):
        # Reset scores and ball positions
        self.left_score = 0
        self.right_score = 0

        self.current_server *= -1

        # ball_1_pos, ball_2_pos = self.get_current_server_pos()
        # self.ball1 = pygame.Rect(*ball_1_pos, BALL_WIDTH, BALL_HEIGHT)
        # self.ball2 = pygame.Rect(*ball_2_pos, BALL_WIDTH, BALL_HEIGHT)
        # self.ball1.topleft, self.ball2.topleft = self.get_current_server_pos()
        self.ball1.topleft = generate_center_position()
        self.ball2.topleft = generate_center_position()

        self.state = START

        self.done = False

        self.left_paddle_1_total_action = 0
        self.left_paddle_2_total_action = 0
        self.right_paddle_1_total_action = 0
        self.right_paddle_2_total_action = 0

        self.reward_right_1 = 0
        self.reward_right_2 = 0
        self.reward_left_1 = 0
        self.reward_left_2 = 0
        # self.ball1_speed_x = 3
        # self.ball1_speed_y = 3
        # self.ball2_speed_x = -3
        # self.ball2_speed_y = -3
        # self.ball1_speed_x = random.choice([-3, 3])
        # self.ball1_speed_y = random.choice([-3, 3])
        # self.ball2_speed_x = random.choice([-3, 3])
        # self.ball2_speed_y = random.choice([-3, 3])
        (self.ball1_speed_x, self.ball1_speed_y), (self.ball2_speed_x, self.ball2_speed_y) = [self.generate_velocity(direction=i) for i in [-1, 1]]

        # counting for each paddel collide with a ball
        self.right_paddle1_count = 0
        self.right_paddle2_count = 0
        self.left_paddle1_count = 0
        self.left_paddle2_count = 0

        # number of out for each ball
        self.ball1_count  = 0
        self.ball2_count  = 0

        # running time for the game
        self.time = 0

        self.left_paddle1_last_moved = 0
        self.left_paddle2_last_moved = 0
        self.right_paddle1_last_moved = 0
        self.right_paddle2_last_moved = 0

        self.left_paddle_last_nonoverlap = 0
        self.right_paddle_last_nonoverlap = 0

        # punishments
        self.left_paddle1_stay = 0
        self.left_paddle1_corner = 0
        
        self.left_paddle2_stay = 0
        self.left_paddle2_corner = 0
        
        self.right_paddle1_stay = 0
        self.right_paddle1_corner = 0
        
        self.right_paddle2_stay = 0
        self.right_paddle2_corner = 0
        
        self.left_overlap = 0
        self.right_overlap = 0
        
        
        self.left_paddle1 = pygame.Rect(5, SCREEN_HEIGHT // 4 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.left_paddle2 = pygame.Rect(5, 3 * SCREEN_HEIGHT // 4 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.right_paddle1 = pygame.Rect(SCREEN_WIDTH - 5 - PADDLE_WIDTH, SCREEN_HEIGHT // 4 - PADDLE_HEIGHT // 2,
                                         PADDLE_WIDTH, PADDLE_HEIGHT)
        self.right_paddle2 = pygame.Rect(SCREEN_WIDTH - 5 - PADDLE_WIDTH, 3 * SCREEN_HEIGHT // 4 - PADDLE_HEIGHT // 2,
                                         PADDLE_WIDTH, PADDLE_HEIGHT)
        

    def get_info(self):
        result = [
            self.right_paddle1_count, self.right_paddle2_count,
            self.left_paddle1_count, self.left_paddle2_count,
            self.ball1_count, self.ball2_count, self.time
        ]
        return result

    def build_state(self):
        # get the coordination for the center of the ball
        ball1_x = self.ball1.centerx
        ball1_y = self.ball1.centery
        
        ball2_x = self.ball2.centerx
        ball2_y = self.ball2.centery

        # paddles' location (coordination)
        left_paddle1_center = self.left_paddle1.centery
        left_paddle1_top = self.left_paddle1.y
        left_paddle1_bottom = self.left_paddle1.bottom
        
        left_paddle2_center = self.left_paddle2.centery
        left_paddle2_top = self.left_paddle2.y
        left_paddle2_bottom = self.left_paddle2.bottom

        right_paddle1_center = self.right_paddle1.centery
        right_paddle1_top = self.right_paddle1.y
        right_paddle1_bottom = self.right_paddle1.bottom

        right_paddle2_center = self.right_paddle2.centery
        right_paddle2_top = self.right_paddle2.y
        right_paddle2_bottom = self.right_paddle2.bottom
        
        # result = [
        #     SCREEN_WIDTH - ball1_x, SCREEN_WIDTH - ball2_x, SCREEN_HEIGHT - ball1_y, SCREEN_HEIGHT - ball2_y,
        #     ball1_x, ball1_y, ball2_x, ball2_y,
        #
        #     self.ball1_speed_x, self.ball1_speed_y, self.ball2_speed_x, self.ball2_speed_y,
        #
        #     left_paddle1_center,
        #     left_paddle1_top - ball1_y, ball1_y - left_paddle1_bottom,
        #     left_paddle1_top - ball2_y, ball2_y - left_paddle1_bottom,
        #
        #     left_paddle2_center,
        #     left_paddle2_top - ball1_y, ball1_y - left_paddle2_bottom,
        #     left_paddle2_top - ball2_y, ball2_y - left_paddle2_bottom,
        #
        #     right_paddle1_center,
        #     right_paddle1_top - ball1_y, ball1_y - right_paddle1_bottom,
        #     right_paddle1_top - ball2_y, ball2_y - right_paddle1_bottom,
        #
        #     right_paddle2_center,
        #     right_paddle2_top - ball1_y, ball1_y - right_paddle2_bottom,
        #     right_paddle2_top - ball2_y, ball2_y - right_paddle2_bottom,
        #     ]

        def divide_SW(n):
            return n/SCREEN_WIDTH

        def divide_HI(n):
            return n/SCREEN_HEIGHT

        def positive(n):
            if n < 0.:
                return 0.
            else:
                return divide_HI(n)

        result = [
            divide_SW(SCREEN_WIDTH - ball1_x), divide_SW(SCREEN_WIDTH - ball2_x),
            divide_HI(SCREEN_HEIGHT - ball1_y), divide_HI(SCREEN_HEIGHT - ball2_y),

            divide_SW(ball1_x), divide_HI(ball1_y), divide_SW(ball2_x), divide_HI(ball2_y),

            self.ball1_speed_x, self.ball1_speed_y, self.ball2_speed_x, self.ball2_speed_y,

            divide_HI(left_paddle1_center),
            positive(left_paddle1_top - ball1_y), positive(ball1_y - left_paddle1_bottom),
            positive(left_paddle1_top - ball2_y), positive(ball2_y - left_paddle1_bottom),

            divide_HI(left_paddle2_center),
            positive(left_paddle2_top - ball1_y), positive(ball1_y - left_paddle2_bottom),
            positive(left_paddle2_top - ball2_y), positive(ball2_y - left_paddle2_bottom),

            divide_HI(right_paddle1_center),
            positive(right_paddle1_top - ball1_y), positive(ball1_y - right_paddle1_bottom),
            positive(right_paddle1_top - ball2_y), positive(ball2_y - right_paddle1_bottom),

            divide_HI(right_paddle2_center),
            positive(right_paddle2_top - ball1_y), positive(ball1_y - right_paddle2_bottom),
            positive(right_paddle2_top - ball2_y), positive(ball2_y - right_paddle2_bottom),
        ]

        result = np.array(result)  # 24
        return result
        
    def step(self, action_right, action_left):
        self.time += 1

        action_right_1, action_right_2 = action_right
        action_left_1, action_left_2 = action_left

        self.left_paddle_1_total_action += 1 if action_left_1 != 0 else 0
        self.left_paddle_2_total_action += 1 if action_left_2 != 0 else 0
        self.right_paddle_1_total_action += 1 if action_right_1 != 0 else 0
        self.right_paddle_2_total_action += 1 if action_right_2 != 0 else 0

        # print(*action_left)
        # print(*action_right)

        left1_reward = 0
        left2_reward = 0
        right1_reward = 0
        right2_reward = 0
        
        
        
        # # Controls for right_paddle1
        # if action_right_1 == 1:
        #     if self.right_paddle1.top > 0:
        #         self.right_paddle1.move_ip(0, -PADDLE_SPEED)
        #     else:
        #         self.reward_right_1 = -0.1
        
        # elif action_right_1 == 2:
        #     if self.right_paddle1.bottom < SCREEN_HEIGHT:
        #         self.right_paddle1.move_ip(0, PADDLE_SPEED)
        #     else:
        #         self.reward_right_1 = -0.1
        
        # # Controls for right_paddle2
        # if action_right_2 == 1:
        #     if self.right_paddle2.top > 0:
        #         self.right_paddle2.move_ip(0, -PADDLE_SPEED)
        #     else:
        #         self.reward_right_2 = -0.1
        
        # elif action_right_2 == 2:
        #     if self.right_paddle2.bottom < SCREEN_HEIGHT:
        #         self.right_paddle2.move_ip(0, PADDLE_SPEED)
        #     else:
        #         self.reward_right_2 = -0.1
        
        # # Controls for left_paddle1
        # if action_left_1 == 1:
        #     if self.left_paddle1.top > 0:
        #         self.left_paddle1.move_ip(0, -PADDLE_SPEED)
        #     else:
        #         self.reward_left_1 = -0.1
        # elif action_left_1 == 2:
        #     if self.left_paddle1.bottom < SCREEN_HEIGHT:
        #         self.left_paddle1.move_ip(0, PADDLE_SPEED)
        #     else:
        #         self.reward_left_1 = -0.1
        
        # # Controls for left_paddle2
        # if action_left_2 == 1:
        #     if self.left_paddle2.top > 0:
        #         self.left_paddle2.move_ip(0, -PADDLE_SPEED)
        #     else:
        #         self.reward_left_2 = -0.1
        
        # elif action_left_2 == 2:
        #     if self.left_paddle2.bottom < SCREEN_HEIGHT:
        #         self.left_paddle2.move_ip(0, PADDLE_SPEED)
        #     else:
        #         self.reward_left_2 = -0.1


        # Controls for right_paddle1

        # save the center_y of the paddles of the last state
        self.last_left_paddle1_y = self.left_paddle1.centery
        self.last_left_paddle2_y = self.left_paddle2.centery
        self.last_right_paddle1_y = self.right_paddle1.centery
        self.last_right_paddle2_y = self.right_paddle2.centery

        if action_right_1 == 1 and self.right_paddle1.top > 0:  # if the upper space is no zero
            self.right_paddle1.move_ip(0, -PADDLE_SPEED)        # paddle move up according to the paddle speed
        elif action_right_1 == 2 and self.right_paddle1.bottom < SCREEN_HEIGHT:  # if the downer space is not zero
            self.right_paddle1.move_ip(0, PADDLE_SPEED)                          # paddle move down according to the paddle speed


        # Controls for right_paddle2
        if action_right_2 == 1 and self.right_paddle2.top > 0:
            self.right_paddle2.move_ip(0, -PADDLE_SPEED)
        if action_right_2 == 2 and self.right_paddle2.bottom < SCREEN_HEIGHT:
            self.right_paddle2.move_ip(0, PADDLE_SPEED)

        # Controls for left_paddle1
        if action_left_1 == 1 and self.left_paddle1.top > 0:
            self.left_paddle1.move_ip(0, -PADDLE_SPEED)
        elif action_left_1 == 2 and self.left_paddle1.bottom < SCREEN_HEIGHT:
            self.left_paddle1.move_ip(0, PADDLE_SPEED)

        # Controls for left_paddle2
        if action_left_2 == 1 and self.left_paddle2.top > 0:
            self.left_paddle2.move_ip(0, -PADDLE_SPEED)
        if action_left_2 == 2 and self.left_paddle2.bottom < SCREEN_HEIGHT:
            self.left_paddle2.move_ip(0, PADDLE_SPEED)

       
        # not encourage the paddel always stay at the corner of the game window
        if self.right_paddle1.top == 0 or self.right_paddle1.bottom == SCREEN_HEIGHT:
            self.reward_right_1 -= CORNER_PUNISHMENT
            right1_reward -= CORNER_PUNISHMENT
            self.right_paddle1_corner += CORNER_PUNISHMENT
            # print("right_1 CORNER PUNISHMENT")

        if self.right_paddle2.top == 0 or self.right_paddle2.bottom == SCREEN_HEIGHT:
            self.reward_right_2 -= CORNER_PUNISHMENT
            right2_reward -= CORNER_PUNISHMENT
            self.right_paddle2_corner += CORNER_PUNISHMENT
            # print("right_2 CORNER PUNISHMENT")
        if self.left_paddle1.top == 0 or self.left_paddle1.bottom == SCREEN_HEIGHT:
            self.reward_left_1 -= CORNER_PUNISHMENT
            left1_reward -= CORNER_PUNISHMENT
            self.left_paddle1_corner += CORNER_PUNISHMENT
            # print("left_1 CORNER PUNISHMENT")
        if self.left_paddle2.top == 0 or self.left_paddle2.bottom == SCREEN_HEIGHT:
            self.reward_left_2 -= CORNER_PUNISHMENT
            left2_reward -= CORNER_PUNISHMENT
            self.left_paddle2_corner += CORNER_PUNISHMENT
            # print("left_2 CORNER PUNISHMENT")
        
        temp = get_punishment_from_staying(self.left_paddle_1_total_action, self.time)
        self.reward_left_1 -= temp
        left1_reward += temp

        temp = get_punishment_from_staying(self.left_paddle_2_total_action, self.time)
        self.reward_left_2 -= temp
        left2_reward += temp

        temp = get_punishment_from_staying(self.right_paddle_1_total_action, self.time)
        self.reward_right_1 -= temp
        right1_reward += temp

        temp = get_punishment_from_staying(self.right_paddle_2_total_action, self.time)
        self.reward_right_2 -= temp
        right2_reward += temp

        self.run()

        # if self.left_paddle1.centery != self.last_left_paddle1_y:
        #     if self.time - self.left_paddle1_last_moved >= STAY_PUNISHMENT_INTERVAL:
        #         self.reward_left_1 -= STAY_PUNISHMENT
        #         print("LEFT 1 STAY PUNISHMENT")
        #     self.left_paddle1_last_moved = self.time
        # if self.left_paddle2.centery != self.last_left_paddle2_y:
        #     if self.time - self.left_paddle2_last_moved >= STAY_PUNISHMENT_INTERVAL:
        #         self.reward_left_2 -= STAY_PUNISHMENT
        #         print("LEFT 2 STAY PUNISHMENT")
        #     self.left_paddle2_last_moved = self.time
        # if self.right_paddle1.centery != self.last_right_paddle1_y:
        #     if self.time - self.right_paddle1_last_moved >= STAY_PUNISHMENT_INTERVAL:
        #         self.reward_right_1 -= STAY_PUNISHMENT
        #         print("RIGHT 1 STAY PUNISHMENT")
        #     self.right_paddle1_last_moved = self.time
        # if self.right_paddle2.centery != self.last_right_paddle2_y:
        #     if self.time - self.right_paddle2_last_moved >= STAY_PUNISHMENT_INTERVAL:
        #         self.reward_right_2 -= STAY_PUNISHMENT
        #         print("RIGHT 2 STAY PUNISHMENT")
        #     self.right_paddle2_last_moved = self.time

        if self.left_paddle1.centery == self.last_left_paddle1_y:
            self.reward_left_1 -= STAY_PUNISHMENT
            left1_reward -= STAY_PUNISHMENT
            self.left_paddle1_stay += STAY_PUNISHMENT
            # print("LEFT 1 STAY PUNISHMENT")
        if self.left_paddle2.centery == self.last_left_paddle2_y:
            self.reward_left_2 -= STAY_PUNISHMENT
            left2_reward -= STAY_PUNISHMENT
            self.left_paddle2_stay += STAY_PUNISHMENT
            # print("LEFT 2 STAY PUNISHMENT")
        if self.right_paddle1.centery == self.last_right_paddle1_y:
            self.reward_right_1 -= STAY_PUNISHMENT
            right1_reward -= STAY_PUNISHMENT
            self.right_paddle1_stay += STAY_PUNISHMENT
            # print("RIGHT 1 STAY PUNISHMENT")
        if self.right_paddle2.centery == self.last_right_paddle2_y:
            self.reward_right_2 -= STAY_PUNISHMENT
            right2_reward -= STAY_PUNISHMENT
            self.right_paddle2_stay += STAY_PUNISHMENT
            # print("RIGHT 2 STAY PUNISHMENT")

        if abs(self.left_paddle1.centery - self.left_paddle2.centery) < PADDLE_HEIGHT:
            self.reward_left_1 -= OVERLAP_PUNISHMENT
            self.reward_left_2 -= OVERLAP_PUNISHMENT
            left1_reward -= OVERLAP_PUNISHMENT
            left2_reward -= OVERLAP_PUNISHMENT
            self.left_overlap += OVERLAP_PUNISHMENT
            # print("LEFT OVERLAP PUNISHMENT")
        if abs(self.right_paddle1.centery - self.right_paddle2.centery) < PADDLE_HEIGHT:
            self.reward_right_1 -= OVERLAP_PUNISHMENT
            self.reward_right_2 -= OVERLAP_PUNISHMENT
            right1_reward -= OVERLAP_PUNISHMENT
            right2_reward -= OVERLAP_PUNISHMENT
            self.right_overlap += OVERLAP_PUNISHMENT
            # print("RIGHT OVERLAP PUNISHMENT")

        temp = get_reward_from_distance(abs(self.right_paddle1.centery - self.ball1.centery),True)
        self.reward_right_1 += temp
        right1_reward += temp
        
        temp = get_reward_from_distance(abs(self.right_paddle1.centery - self.ball2.centery),False)
        self.reward_right_1 += temp
        right1_reward += temp

        temp = get_reward_from_distance(abs(self.right_paddle2.centery - self.ball1.centery),False)
        self.reward_right_2 += temp
        right2_reward += temp
        
        temp = get_reward_from_distance(abs(self.right_paddle2.centery - self.ball2.centery),True)
        self.reward_right_2 += temp
        right2_reward += temp

        temp = get_reward_from_distance(abs(self.left_paddle1.centery - self.ball1.centery),True)
        self.reward_left_1 += temp
        left1_reward += temp
        
        temp = get_reward_from_distance(abs(self.left_paddle1.centery - self.ball2.centery),False)
        self.reward_left_1 += temp
        left1_reward += temp
        
        temp = get_reward_from_distance(abs(self.left_paddle2.centery - self.ball1.centery),False)
        self.reward_left_2 += temp
        left2_reward += temp

        temp = get_reward_from_distance(abs(self.left_paddle2.centery - self.ball2.centery),True)
        self.reward_left_2 += temp
        left2_reward += temp
        

        # if abs(self.left_paddle1.centery - self.left_paddle2.centery) > PADDLE_HEIGHT:
        #     if self.time - self.left_paddle_last_nonoverlap >= OVERLAP_PUNISHMENT_INTERVAL:
        #         self.reward_left_1 -= OVERLAP_PUNISHMENT
        #         self.reward_left_2 -= OVERLAP_PUNISHMENT
        #         print("LEFT OVERLAP PUNISHMENT")
        #     self.left_paddle_last_nonoverlap = self.time
        # if abs(self.right_paddle1.centery - self.right_paddle2.centery) > PADDLE_HEIGHT:
        #     if self.time - self.right_paddle_last_nonoverlap >= OVERLAP_PUNISHMENT_INTERVAL:
        #         self.reward_right_1 -= OVERLAP_PUNISHMENT
        #         self.reward_right_2 -= OVERLAP_PUNISHMENT
        #         print("RIGHT OVERLAP PUNISHMENT")
        #     self.right_paddle_last_nonoverlap = self.time
        
        # #transfer the value of reward and reset the value of the self reward 
        # if self.reward_right_1 != 0:
        #     reward_right_1 = self.reward_right_1

        # if self.reward_right_2 != 0:
        #     reward_right_2 = self.reward_right_2

        # if self.reward_left_1 != 0:
        #     reward_left_1 = self.reward_left_1

        # if self.reward_left_2 != 0:
        #     reward_left_2 = self.reward_left_2

        # back to the original state
        state = self.build_state()

        # self.done = 1

        return state, left1_reward, left2_reward, right1_reward, right2_reward, self.done


    def start(self):
        while True:
            if self.state == START:
                self.handle_start()
            elif self.state == PLAYING:
                self.handle_playing()
            elif self.state == PAUSED:
                self.handle_paused()
            elif self.state == GAME_OVER:
                self.handle_game_over()

    def run(self):
        if self.state == START:
            self.handle_start()
        elif self.state == PLAYING:
            self.handle_playing()
        elif self.state == PAUSED:
            self.handle_paused()
        elif self.state == GAME_OVER:
            self.handle_game_over()

    def handle_start(self):
        # Display start screen and wait for player to press a key to start
        # Transition to PLAYING state when ready
        self.state = PLAYING

    def handle_playing(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                exit()


        if HUMAN_CONTROL:
            keys = pygame.key.get_pressed()
            # Controls for left_paddle1
            if keys[K_w] and self.left_paddle1.top > 0:
                self.left_paddle1.move_ip(0, -PADDLE_SPEED)    # press w move up
            if keys[K_s] and self.left_paddle1.bottom < SCREEN_HEIGHT:  
                self.left_paddle1.move_ip(0, PADDLE_SPEED)     # press s move down

            # Controls for left_paddle2
            if keys[K_a] and self.left_paddle2.top > 0:
                self.left_paddle2.move_ip(0, -PADDLE_SPEED)   # press a move up
            if keys[K_d] and self.left_paddle2.bottom < SCREEN_HEIGHT:  
                self.left_paddle2.move_ip(0, PADDLE_SPEED)    # press d move down

            # Controls for right_paddle1
            if keys[K_UP] and self.right_paddle1.top > 0:
                self.right_paddle1.move_ip(0, -PADDLE_SPEED)  # press up move up
            if keys[K_DOWN] and self.right_paddle1.bottom < SCREEN_HEIGHT:
                self.right_paddle1.move_ip(0, PADDLE_SPEED)   # press down move down

            # Controls for right_paddle2
            if keys[K_LEFT] and self.right_paddle2.top > 0:
                self.right_paddle2.move_ip(0, -PADDLE_SPEED)  # press left move up
            if keys[K_RIGHT] and self.right_paddle2.bottom < SCREEN_HEIGHT:
                self.right_paddle2.move_ip(0, PADDLE_SPEED)   # press right move down

            if keys[K_t]:
                print('state')
                print(self.build_state())
                print(len(self.build_state()))

            if keys[K_r]:
                print('reset')
                self.reset()

        # Move the balls and check for collisions with walls and paddles
        self.move_and_collide(self.ball1, self.ball1_speed_x, self.ball1_speed_y, ball_name='ball1')
        self.move_and_collide(self.ball2, self.ball2_speed_x, self.ball2_speed_y, ball_name='ball2')

        # if self.left_score >= 10 or self.right_score >= 10:
        # if self.left_score >= 3 or self.right_score >= 3:
        if self.left_score >= MAX_SCORE or self.right_score >= MAX_SCORE:
            self.state = GAME_OVER

        if RENDER:
            # Draw everything
            self.screen.fill(BLACK)
            pygame.draw.rect(self.screen, RED, self.left_paddle1)
            pygame.draw.rect(self.screen, GREEN, self.left_paddle2)
            pygame.draw.rect(self.screen, RED, self.right_paddle1)
            pygame.draw.rect(self.screen, GREEN, self.right_paddle2)
            pygame.draw.ellipse(self.screen, RED, self.ball1)
            pygame.draw.ellipse(self.screen, GREEN, self.ball2)
            pygame.draw.aaline(self.screen, WHITE, (SCREEN_WIDTH // 2, 0), (SCREEN_WIDTH // 2, SCREEN_HEIGHT))

            # Display scores
            left_score_text = self.font.render(str(self.left_score), True, WHITE)
            right_score_text = self.font.render(str(self.right_score), True, WHITE)
            self.screen.blit(left_score_text, (SCREEN_WIDTH // 4, 10))   # make the text or number shown overlap on the figure or picture
            self.screen.blit(right_score_text, (3 * SCREEN_WIDTH // 4, 10))

            pygame.display.flip()
            if LIMIT_SPEED:
                pygame.time.Clock().tick()


    def normalise(self, x, y, target):
        mag = np.sqrt(x ** 2 + y ** 2)
        unit_x = x / mag
        unit_y = y / mag
        return target * unit_x, target * unit_y

        
    def modify_vel(self, ball, ball_name, last_paddle_y, current_paddle_y):  # randomise the velocity after each collision
        if last_paddle_y == current_paddle_y:
            return
        if ball_name == "ball1":
            vel_x = self.ball1_speed_x
            vel_y = self.ball1_speed_y
        else:
            vel_x = self.ball2_speed_x
            vel_y = self.ball2_speed_y
        magnitude = np.sqrt(vel_x ** 2 + vel_y ** 2)
        delta = random.uniform(*Y_RANDOM_RANGE)
        if current_paddle_y < last_paddle_y:
            vel_y -= delta
        else:
            vel_y += delta
        if ball_name == "ball1":
            self.ball1_speed_x, self.ball1_speed_y = self.normalise(vel_x, vel_y, magnitude)
        else:
            self.ball2_speed_x, self.ball2_speed_y = self.normalise(vel_x, vel_y, magnitude)


    def move_and_collide(self, ball, ball_speed_x, ball_speed_y, ball_name=None):
        ball.move_ip(ball_speed_x, ball_speed_y)

        # when the ball touch the boundary then rebounce
        if ball.top <= 0 or ball.bottom >= SCREEN_HEIGHT:
            if ball_name == 'ball1':
                self.ball1_speed_y = -self.ball1_speed_y
            else:
                self.ball2_speed_y = -self.ball2_speed_y
        

        

        # score collection
        if ball.left <= 0:
            # self.reward_left_1 = -1
            # self.reward_left_2 = -1
            if ball_name == 'ball1':
                self.ball1_count += 1
                # self.reward_left_1 -= MISS_CORRECT_BALL_PUNISHMENT
                # print("left_1 MISS_correct_ball PUNISHMENT")
                # self.reward_left_2 -= MISS_INCORRECT_BALL_PUNISHMENT
                # print("left_2 MISS_incorrect_ball PUNISHMENT")
                # ball 1
                self.right_score += 1
                # reserve from the middle
                ball.topleft = self.left_paddle1.center
                # self.ball1_speed_x = 3 + random.uniform(-1, 1)  # Serve to the right
                # self.ball1_speed_y = random.choice([-3, 3])
                self.ball1_speed_x, self.ball1_speed_y = self.generate_velocity(direction=RIGHT)
                # self.ball1_speed_x = 3
                # self.ball1_speed_y = 3


            else:
                self.ball2_count += 1
                # self.reward_left_1 -= MISS_INCORRECT_BALL_PUNISHMENT
                # print("left_1 MISS_incorrect_ball PUNISHMENT")
                # self.reward_left_2 -= MISS_CORRECT_BALL_PUNISHMENT
                # print("left_2 MISS_correct_ball PUNISHMENT")
                # ball 2
                self.right_score += 1
                ball.topleft = self.left_paddle2.center
                # self.ball2_speed_x = random.choice([-3, 3]) + random.uniform(-1, 1)
                # self.ball2_speed_y = random.choice([-3, 3])
                (self.ball2_speed_x, self.ball2_speed_y) = self.generate_velocity(direction=RIGHT)
                # self.ball2_speed_x = -3
                # self.ball2_speed_y = -3



        # collect the score

        if ball.right >= SCREEN_WIDTH:
            # self.reward_right_1 = -1
            # self.reward_right_2 = -1
            if ball_name == 'ball1':
                self.ball1_count += 1
                # self.reward_right_1 -= MISS_CORRECT_BALL_PUNISHMENT
                # print("right_1 MISS_correct_ball PUNISHMENT")
                # self.reward_right_2 -= MISS_INCORRECT_BALL_PUNISHMENT
                # print("right_2 MISS_incorrect_ball PUNISHMENT")
                self.left_score += 1
                ball.topright = self.right_paddle1.center
                # self.ball1_speed_x = -3 + random.uniform(-1, 1)  # Serve to the left
                # self.ball1_speed_y = random.choice([-3, 3])
                self.ball1_speed_x, self.ball1_speed_y = self.generate_velocity(direction=LEFT)


            else:
                self.ball2_count += 1
                # self.reward_right_1 -= MISS_INCORRECT_BALL_PUNISHMENT
                # print("right_1 MISS_incorrect_ball PUNISHMENT")
                # self.reward_right_2 -= MISS_CORRECT_BALL_PUNISHMENT
                # print("right_2 MISS_correct_ball PUNISHMENT")
                self.left_score += 1
                ball.topright = self.right_paddle2.center
                # self.ball2_speed_x = random.choice([-3, 3]) + random.uniform(-1, 1)
                # self.ball2_speed_y = random.choice([-3, 3])
                self.ball2_speed_x, self.ball2_speed_y = self.generate_velocity(direction=LEFT)

        paddles = [self.left_paddle1, self.left_paddle2, self.right_paddle1, self.right_paddle2]
        for i, paddle in enumerate(paddles):
            if i == 0:
                last_y = self.last_left_paddle1_y
                current_y = self.left_paddle1.centery
            elif i == 1:
                last_y = self.last_left_paddle2_y
                current_y = self.left_paddle1.centery
            elif i == 2:
                last_y = self.last_right_paddle1_y
                current_y = self.right_paddle1.centery
            elif i == 4:
                last_y = self.last_right_paddle2_y
                current_y = self.right_paddle2.centery
            if ball.colliderect(paddle):
                if ball_name == 'ball1':
                    self.modify_vel(ball, ball_name, last_y, current_y)
                    self.ball1_speed_x = -self.ball1_speed_x
                    # Check which side of the paddle the ball hit
                    if ball.right > paddle.left and ball.left < paddle.left:
                        # Ball hit left side of paddle,
                        # successfully rebounce the ball for the right side
                        if i == 3:
                            self.reward_right_1 += CORRECT_BALL_REWARD
                            self.right_paddle1_count += 1
                        else:
                            self.reward_right_2 += INCORRECT_BALL_REWARD
                            # self.reward_right_1 -= INCORRECT_BALL_PUNISHMENT
                            # print("right_1 incorrect_ball PUNISHMENT")
                            self.right_paddle2_count += 1
                        ball.right = paddle.left
                    elif ball.left < paddle.right and ball.right > paddle.right:
                        # Ball hit right side of paddle
                        # the left sufccessfully rebounce the ball
                        if i == 0:
                            self.reward_left_1 += CORRECT_BALL_REWARD
                            self.left_paddle1_count += 1
                        else:
                            self.reward_left_2 += INCORRECT_BALL_REWARD
                            # self.reward_left_1 -= INCORRECT_BALL_PUNISHMENT
                            # print("left_1 incorrect_ball PUNISHMENT")
                            self.left_paddle2_count += 1
                        ball.left = paddle.right
                    else:
                        self.ball1_speed_y = -self.ball1_speed_y  # Ball hit top or bottom of paddle
                else:
                    self.modify_vel(ball, ball_name, last_y, current_y)
                    self.ball2_speed_x = -self.ball2_speed_x
                    # Check which side of the paddle the ball hit
                    if ball.right > paddle.left and ball.left < paddle.left:
                        # Ball hit left side of paddle
                        # the right side successfully rebounce back
                        if i == 3:
                            self.reward_right_1 += INCORRECT_BALL_REWARD
                            # self.reward_right_2 -= INCORRECT_BALL_PUNISHMENT
                            # print("right_2 incorrect_ball PUNISHMENT")
                            self.right_paddle1_count += 1
                        else:
                            self.reward_right_2 += CORRECT_BALL_REWARD
                            self.right_paddle2_count += 1
                        ball.right = paddle.left
                    elif ball.left < paddle.right and ball.right > paddle.right:
                        # Ball hit right side of paddle
                        # the left side successsfully rebounce
                        if i == 0:
                            self.reward_left_1 += INCORRECT_BALL_REWARD
                            # self.reward_left_2 -= INCORRECT_BALL_PUNISHMENT
                            # print("left_2 incorrect_ball PUNISHMENT")
                            self.left_paddle1_count += 1
                        else:
                            self.reward_left_2 += CORRECT_BALL_REWARD
                            self.left_paddle2_count += 1
                        ball.left = paddle.right
                    else:
                        self.ball2_speed_y = -self.ball2_speed_y  # Ball hit top or bottom of paddle

        if ball.colliderect(self.left_paddle1) or ball.colliderect(self.left_paddle2) or ball.colliderect(
                self.right_paddle1) or ball.colliderect(self.right_paddle2):
            if ball_name == 'ball1':
                self.ball1_speed_x *= -1
                self.ball1_speed_y *= random.uniform(0, 2)
            else:
                self.ball2_speed_x *= -1
                self.ball2_speed_y *= random.uniform(0, 2)


    def handle_paused(self):
        # Display paused screen and wait for player to resume or quit
        pass

    def handle_game_over(self):
        self.done = True
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                exit()

        keys = pygame.key.get_pressed()
        if keys[K_r]:
            print('reset')
            self.reset()

        # Draw everything
        self.screen.fill(BLACK)
        game_over_text = self.font.render("GAME OVER", True, WHITE)
        game_over_pos = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        self.screen.blit(game_over_text, game_over_pos)

        # Show scores
        left_score_text = self.font.render(str(self.left_score), True, WHITE)
        right_score_text = self.font.render(str(self.right_score), True, WHITE)
        left_score_pos = left_score_text.get_rect(center=(SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2 + 50))
        right_score_pos = right_score_text.get_rect(center=(3 * SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2 + 50))
        self.screen.blit(left_score_text, left_score_pos)
        self.screen.blit(right_score_text, right_score_pos)

        pygame.display.flip()
        if LIMIT_SPEED:
            pygame.time.Clock().tick(60)


    def save_screen(self, path='./screen.png'):
        pygame.image.save(self.screen, path)

    def get_screen_data(self):
        data = pygame.image.tostring(self.screen, 'RGB')
        data = np.frombuffer(data, dtype=np.uint8)
        data = data.reshape((SCREEN_HEIGHT, SCREEN_WIDTH, 3))

        return data


if __name__ == '__main__':

    game = PongGame()
    game.start()