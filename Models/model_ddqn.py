import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=32, fc2_units=16):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc12 = nn.Linear(fc1_units, fc2_units)
        
        # Value Stream
        self.fc2 = nn.Linear(fc2_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        
        # Advantage Stream
        self.fc2bis = nn.Linear(fc2_units, fc2_units)
        self.fc3bis = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc12(x))
        
        # VALUE STREAM
        xvalue = F.relu(self.fc2(x))
        xvalue = F.relu(self.fc3(xvalue))
        
        # ADVANTAGE STREAM
        xadvantage = F.relu(self.fc2bis(x))
        xadvantage = F.relu(self.fc3bis(xadvantage))
        
        qvals = xvalue + (xadvantage - xadvantage.mean())
        
        return qvals
