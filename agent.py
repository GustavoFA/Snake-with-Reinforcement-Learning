import torch
import random
import numpy as np
from collections import deque # store the memory
from sneak_game import SnakeGame, Direction, Point
from model import LinearQNet, QTrainer
from plots import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001 #1e-3

class Agent:

    def __init__(self):
        
        # control the number of games
        self.n_games = 0
        
        # randomness
        self.epsilon = 0 
        # discount rate (must be smaller than 1)
        self.gamma = 0.9

        # control a memory model. If fully the memomry -> popleft()
        self.memory = deque(maxlen=MAX_MEMORY)

        # model
        self.model = LinearQNet()

        # trainer
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game) -> np.array:
        """
        
        """
        
        head = game.snake[0]

        point_l = Point(head.x - game.block_size, head.y)
        point_r = Point(head.x + game.block_size, head.y)
        point_u = Point(head.x, head.y - game.block_size)
        point_d = Point(head.x, head.y + game.block_size)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),
            
            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x, # food on left
            game.food.x > game.head.x, # food on right
            game.food.y < game.head.y, # food up
            game.food.y > game.head.y  # food down 
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self) -> None:
        if len(self.memory ) > BATCH_SIZE:
            # list of tuples
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done) -> None:
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state) -> list:
        # random moves: tradeoff exploration | exploitation
        # self.epsilon = 80 - self.n_games
        self.epsilon = max(0, 80 - self.n_games)
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0) 
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move

def train() -> None:
    """
    
    """
    plot_scores = []
    plot_mean_scores = []
    total_scores = 0
    record = 0
    agent = Agent()
    game = SnakeGame()

    while True:
        # get the old state
        state_old = agent.get_state(game)

        # get the move
        final_move = agent.get_action(state_old)

        # perform the move and get the new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory and plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
            
            print('Game', agent.n_games, 'Score', score, 'Record', record)

            plot_scores.append(score)
            total_scores += score
            mean_score = total_scores / agent.n_games
            plot_mean_scores.append(mean_score)
            
            plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    train()
