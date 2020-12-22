#### Import sub-modules of the library ####
from .datasets import *
from .models import *
from .utils import *

#### Main classes of the library ####

# Imports for the classes
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F


class FederatedModel:
	def __init__(	self, 
					model = NN(input_dim=10, output_dim=2), 
					model_type = 'nn', 
					fed_optimizer = 'fed_avg', 
					fed_opt_params = {},  
					local_optimizer = 'Adam', 
					local_opt_params = {'lr': 0.01, 'batches':64, 'epochs':4},
					security = None,
					security_params = {}):
		self.model = model
		self.model_type = model_type
		self.fed_optimizer = fed_optimizer
		self.fed_opt_params = fed_opt_params
		self.local_optimizer = local_optimizer
		self.security = security
		self.security_params = security_params

	def state_dict(self):
		# State dict of the current weights and biases the the network 
		model_dict = self.model.state_dict()
		return model_dict

	def load_state_dict(self, model_dict):
		self.model.load_state_dict(model_dict)

	def server_agregate(self, client_weights, client_lens):
		if self.fed_optimizer == 'fed_avg':
			"""
			This function has aggregation method 'wmean'
			wmean takes the weighted mean of the weights of models
			"""
			total = sum(client_lens)
			n     = len(client_weights)
			global_dict = self.state_dict()
			for k in global_dict.keys():
				global_dict[k] = torch.stack([client_weights[i][k].float()*(n*client_lens[i]/total) for i in range(n)], 0).mean(0)
			self.load_state_dict(global_dict)


class LocalModel:
	def __init__(	self, 
					model, 
					model_type = 'nn', 
					local_optimizer = 'Adam', 
					local_opt_params = {'lr': 0.01, 'batch_size':64, 'epochs':4},
					security = None,
					security_params = {}):
		self.model = model
		self.model_type = model_type
		self.optimizer_name = local_optimizer
		self.local_opt_params = local_opt_params
		self.security = security
		self.security_params = security_params
		self.optimizer = self.__generate_opt(self.optimizer_name, **self.local_opt_params)
		self.loss_fn   = F.nll_loss

	def __call__(self, data):
		self.model.eval()
		output = self.model(data)
		return output

	def eval(self, data, prob=False):
		self.model.eval()
		predict_prob = self.model(data)
		predict = output.argmax(dim=1, keepdim=True)
		if prob == False:
			return predict
		else:
			return predict_prob, predict

	def predict(self, data):
		self.model.eval()
		predict_prob = self.model(data)
		predict = output.argmax(dim=1, keepdim=True)
		return predict

	def test(self, data_loader, device='cpu'):
		self.model.eval()
		self.model.to(device)
		test_loss = 0
		correct = 0

		with torch.no_grad():
			for data, target in data_loader:
				data, target = data.to(device), target.to(device)
				output = self.model(data)
				test_loss += self.loss_fn(output, target, reduction='sum').item()  # sum up batch loss
				pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
				correct += pred.eq(target.view_as(pred)).sum().item()

		test_loss /= len(data_loader.dataset)
		acc = correct / len(data_loader.dataset)

		return acc, test_loss

	def train(self, data_loader, device='cpu'):
		self.model.train()
		self.model.to(device)
		epochs = self.local_opt_params['epochs']
		for e in range(epochs):
			for batch_idx, (data, target) in enumerate(data_loader):
				data, target = data.to(device), target.to(device)
				self.optimizer.zero_grad()
				output = self.model(data)
				loss   = self.loss_fn(output, target)
				loss.backward()
				self.optimizer.step()
		return loss.item()

	def state_dict(self):
		# State dict of the current weights and biases the the network 
		model_dict = self.model.state_dict()
		return model_dict

	def load_state_dict(self, model_dict):
		self.model.load_state_dict(model_dict)

	def __generate_opt(self, name='Adam', **kwargs):
		lr = kwargs['lr']
		if name == 'Adam':
			optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

		elif name == 'SGD':
			optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

		elif name == 'RMSprop':
			optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr)

		else:
			raise Exception('Unsupported optimizer name. Supported: Adam, SGD, RMSProp')

		return optimizer