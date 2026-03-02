import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

class SnakeGame:
    """
    Snake game environment.

    This class supports both:
    - Human-controlled gameplay 
    - AI-controller gameplay

    It is suitable for reinforcement learning experiments.

    Args:
        w (int): Width of the game window in pixels.
        h (int): Height of the game window in pixels.
        human (bool): If True, enables keyboard control. If False, expects AI actions.
    """

    def __init__(self, w:int=640, h:int=480, human:bool=False):

        pygame.init()
        self.font = pygame.font.SysFont('arial', 25)
        
        # Sneak square size
        self.block_size = 20
        
        # Identify the user (Human or AI agent)
        self.human = human

        # game screen size
        self.w = w
        self.h = h
        
        # select the sneak speed
        self.speed = 20 if self.human else 40

        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        self.reset()


    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w//2, self.h//2)
        self.snake = [self.head,
                      Point(self.head.x-self.block_size, self.head.y),
                      Point(self.head.x-(2*self.block_size), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0


    def _place_food(self):
        x = random.randint(0, (self.w-self.block_size )//self.block_size )*self.block_size
        y = random.randint(0, (self.h-self.block_size )//self.block_size )*self.block_size
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action:np.array=None):
        """
        
        """

        self.frame_iteration += 1 # control the number of agent iterations 

        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if self.human and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN
        
        # 2. move
        if self.human:
            self._move()
        else:
            self._move(action) # update the head

        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or (self.frame_iteration > 100*len(self.snake) and not self.human):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(self.speed)
        # 6. return game over and score
        return reward, game_over, self.score


    def is_collision(self, pt=None):
        """
        Detect snake collision

        Args:
            pt: 
        """
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - self.block_size or pt.x < 0 or pt.y > self.h - self.block_size or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False


    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, self.block_size, self.block_size))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, self.block_size, self.block_size))

        text = self.font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def _move(self, action:np.array=None) -> None:
        """
        Update the sneak position.

        Args:
            action (np.array | None): For AI agent is spected an array of the action. For the 
            human user is not spected (None).
        """
        # action positions = [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if not self.human:
            if np.array_equal(action, [1, 0, 0]):
                new_dir = clock_wise[idx] # no change
            elif np.array_equal(action, [0, 1, 0]):
                next_idx = (idx + 1) % 4
                new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
            else: # [0, 0, 1]
                next_idx = (idx - 1) % 4
                new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

            self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += self.block_size
        elif self.direction == Direction.LEFT:
            x -= self.block_size
        elif self.direction == Direction.DOWN:
            y += self.block_size
        elif self.direction == Direction.UP:
            y -= self.block_size

        self.head = Point(x, y)


if __name__ == "__main__":
    game = SnakeGame(human=True)

    while True:
        _, game_over, score = game.play_step()

        if game_over == True:
            break

    print(f'Final score = {score}')
