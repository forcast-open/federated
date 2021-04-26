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

class CNN_MNIST(nn.Module):
	def __init__(self):
		super(CNN_MNIST, self).__init__()
		self.conv1 = nn.Conv2d( 1, 64, kernel_size=3)
		self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
		self.pool1 = nn.MaxPool2d(2, 2)
		self.bn1   = nn.BatchNorm2d(64)
		self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
		self.conv4 = nn.Conv2d(128, 128, kernel_size=3)
		self.pool2 = nn.MaxPool2d(2, 2)
		# flatten
		self.bn2   = nn.BatchNorm1d(128 * 4 * 4)
		self.fc1   = nn.Linear(128 * 4 * 4, 512)
		self.fc2   = nn.Linear(512, 10)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = self.pool1(x)
		x = self.bn1(x)
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = self.pool2(x)
		x = x.view(-1, 128 * 4 * 4)
		x = self.bn2(x)
		x = F.relu(self.fc1(x))
		x = F.softmax(self.fc2(x), dim=1)
		
		return x

	def init_weights(self, init_seed=0):
		torch.manual_seed(init_seed)
		for m in self.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
		
		return self