import torch
import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):
    # define nn
    def __init__(self, input_dim, output_dim):
        super(NN, self).__init__()
        self.layer1 = nn.Linear(input_dim,10)
        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, output_dim)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1) # To check with the loss function
        return x