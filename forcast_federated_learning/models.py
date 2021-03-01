import torch
import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):
	# define nn
	def __init__(self, input_dim: int, output_dim: int, hidden_dim = 5, init_seed = 0):
		super(NN, self).__init__()
		self.layer1 = nn.Linear(input_dim,hidden_dim)
		self.layer2 = nn.Linear(hidden_dim, output_dim)

		self.init_weights(init_seed)
		
	def forward(self, x):
		x = F.relu(self.layer1(x))
		x = F.softmax(self.layer2(x), dim=1) # To check with the loss function
		
		return x

	def init_weights(self, init_seed=0):
		torch.manual_seed(init_seed)
		for m in self.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
		
		return self

class NN_REG(nn.Module):
	# define nn
	def __init__(self, input_dim: int, output_dim: int, hidden_dim = 5, init_seed = 0):
		super(NN_REG, self).__init__()
		self.layer1 = nn.Linear(input_dim, hidden_dim)
		self.layer2 = nn.Linear(hidden_dim, output_dim)

		self.init_weights(init_seed)
		
	def forward(self, x):
		x = F.relu(self.layer1(x))
		x = F.relu(self.layer2(x))

		return x

	def init_weights(self, init_seed=0):
		torch.manual_seed(init_seed)
		for m in self.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
		
		return self