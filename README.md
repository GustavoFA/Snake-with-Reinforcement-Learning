# Snake with Reinforcement Learning

In this project we'll train a model with RL (using PyTorch) and apply it on a Pygame environment. I'm following this [case](https://www.youtube.com/playlist?list=PLqnslRFeH2UrDh7vUmJ60YrmWd64mTTKV). For more details go to the [main project](https://github.com/patrickloeber/snake-ai-pytorch).

RL is teaching a software agent how to behave in an environment by telling it how good it's doing. In this project we'll use Deep Q Learning.

## Concepts

### Action Space

The agent has 3 possible actions, how it shows in table below.

| Action      | Meaning       |
| ----------- | ------------- |
| `[1, 0, 0]` | Move straight |
| `[0, 1, 0]` | Turn right    |
| `[0, 0, 1]` | Turn left     |

### States Representations

The state is a vector of 11 features:

- Danger:
    - danger straight
    - danger right
    - danger left
- Direction:
    - moving left/right/up/down
- Food location:
    - food left/right/up/down

### Model

The model is a simple feedforward neural network:

- Input layer: 11 neurons
- Hidden layer: 256 neurons (ReLU)
- Output layer: 3 neurons (Q-values for each action)

### [Deep Q Learning](https://www.freecodecamp.org/news/diving-deeper-into-reinforcement-learning-with-q-learning-c18d0db58efe/)

Q Value = Quality of Action 

Q Value (or Q-table) is a table with m columns, number of actions, and n rows, number of states. Init Q value (init model = random value), then choose action (model.predict(state)) and perform the action; Measure reward and update Q value (+train model). Repeat this steps (training loop).

We've an exploration rate, called epsilon, which we set to 1 in the beginning, because we don't know anything about the values in Q-table, i.e., we need to do a lot of exploration, by randomly choosing the actions. When this number is low we'll do exploration, which means we use already know to select the best action at each step. The idea is big epsilon at the beginning of the training of the Q-function, then, reduce it progressively as the agent becomes more confident at estimating Q-values.

Q-learning is a value-based Reinforcement Learning algorithm that is used to find the optimal action-selection policy using a Q function. It evaluates which action to take based on an action-value function that determines the value of being in a certain state and taking a certain action at that state. The Q function can be estimated using Q-learning, which iteratively updates Q(s,a) using the Bellman Equation.

#### Bellman Equation 

The equation below, called *Bellman equation*, is to solve the Q values.

$$Q_{new}(s, a) = Q(s, a) + \alpha[R(s,a) + \gamma \ maxQ'(s',a') - Q(s,a)]$$

where, $a$ is the action, $s$ is the state, $\alpha$ is the learning rate,  $\gamma$ is the discount rate and $R$ is the reward for taking that action at that state. So, the equation means that the new Q value for that state and that action depends of the current Q value, the reward for taking that action and state and the maximum expected future reward given the new state and all possible actions at that new state.

$Q = model.predict(state_0)$

$Q_{new} = R + \gamma \times max(Q_{state_1})$

Loss = $(Q_{new} - Q)²$ [Mean Square Loss]


### Reward

The agent learns to play Snake by interacting with the environment and receiving rewards:

- +10 → eating food
- -10 → dying

The goal is to maximize cumulative reward over time.

## Requirements
- Pygame
- PyTorch (torch and torchvision)
- MatPlotLib
- iPython

## How to Run
Soon

## Results
Soon



