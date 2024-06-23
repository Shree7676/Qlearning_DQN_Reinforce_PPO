# Import:
# -------
import random

import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# Deep Q-Network:
# ---------------
class Qnet(nn.Module):
    def __init__(self, num_actions, num_states):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(num_states, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_actions)

    def forward(self, input_state):
        layer1_op = F.relu(self.fc1(input_state))
        layer2_op = F.relu(self.fc2(layer1_op))
        layer3_op = self.fc3(layer2_op)
        return layer3_op

    def sample_action(self, input_state, epsilon):
        action_arr = self.forward(input_state)

        if random.random() < epsilon:
            return random.randint(0, 3)
        else:
            return action_arr.argmax().item()
