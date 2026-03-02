import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from datetime import datetime

class LinearQNet(nn.Module):
    """
    Feedforward neural network used to approximate Q-Values.

    Architecture:
        Input : 11 (state_size)
        Linear layer : 256 (hidden_size) + ReLU
        Linear layer : 3 (output_size) 

    Output:
        A vector of Q-values for each possible action:
        [Q(straight), Q(right), Q(left)]
    """

    def __init__(self, input_size:int=11, hidden_size:int=256, output_size:int=3):
        """
        Initialize the neural network.

        Args:
            input_size (int): Number of state features.
            hidden_size (int): Number of neurons in hidden layer.
            output_size (int): Number of possible actions.
        """
        super().__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (Tensor): Input state tensor.

        Returns:
            Tensor: Predicted Q-values for each action.
        """
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name:str=None):
        """
        Save the model weights.

        Args:
            file_name (str, None): Name of the file (must end with .pth).
        """
        model_folder_path = './model_saved'
        os.makedirs(model_folder_path, exist_ok=True)

        if file_name is None:
            file_name = f'model_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.pth'

        file_name = os.path.join(model_folder_path, file_name)

        torch.save(self.state_dict(), file_name)

class QTrainer:
    """
    
    """
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optim = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done) -> tuple:
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)

        if len(state.shape) == 1:
            # (1, x) -> (x, )
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            done = (done, ) # a tuple with one value

        # Predict Q values with current state
        pred = self.model(state)

        # Compute Qnew (only when it's not done)
        target = pred.clone()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma *torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
        
        self.optim.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optim.step()