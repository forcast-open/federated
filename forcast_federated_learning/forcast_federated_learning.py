#### Import sub-modules of the library ####
from .datasets import *
from .models import *
from .utils import *
from .data import *
from .security import *
from .optim import *
from .encryption import *

#### Main classes of the library ####

# Imports for the classes
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus import PrivacyEngine


class FederatedModel:
	def __init__(	self, 
					model, 
					model_type           = 'nn', 
					fed_optimizer        = 'fed_avg', 
					fed_optimizer_params = {}	):

		self.model                 = model
		self.model_type            = model_type
		self.fed_optimizer         = fed_optimizer
		self.fed_optimizer_params  = fed_optimizer_params

	def eval(self):
		return self.model.eval()

	def train(self):
		return self.model.train()

	def to(self, device='cpu'):
		return self.model.to(device)

	def state_dict(self):
		# State dict of the current weights and biases the the network 
		return self.model.state_dict()

	def load_state_dict(self, model_dict):
		return self.model.load_state_dict(model_dict)

	def __call__(self, data):
		self.eval()
		output = self.model(data)
		return output

	def predict(self, data, prob=False):
		self.eval()
		predict_prob = self.model(data)
		predict = output.argmax(dim=1, keepdim=True)
		if prob == False:
			return predict
		else:
			return predict_prob, predict
	
	def server_agregate(self, client_weights, client_lens, secret_key=None):
		if self.fed_optimizer == 'fed_avg':
			"""
			This function has aggregation method 'wmean'
			wmean takes the weighted mean of the weights of models
			"""
			total = sum(client_lens)
			n     = len(client_weights)
			global_dict = self.state_dict()

			if isinstance(client_weights[0], EncStateDict):
				assert (secret_key is not None), 'secret_key needs to be given as a parameter to agregate EncStateDicts.'
				enc_state_dict = 0
				for enc_client_weight, client_len in zip(client_weights, client_lens):
					enc_state_dict = enc_state_dict + (client_len/total) * enc_client_weight
				# decrypt only the result
				state_dict = enc_state_dict.decrypt(secret_key)
				self.load_state_dict(state_dict)


			else:
				for k in global_dict.keys():
					global_dict[k] = torch.stack([client_weights[i][k].float()*(n*client_lens[i]/total) for i in range(n)], 0).mean(0)
				self.load_state_dict(global_dict)

class LocalModel:
	def __init__(	self, 
					model, 
					model_type       = 'nn',
					loss_fn          = nn.CrossEntropyLoss(),
					optimizer_name   = 'Adam', 
					optimizer_params = {'lr': 0.01},
					train_params     = {'epochs': 4},
					security         = None,
					security_params  = {'noise_multiplier': 0.3, 'max_grad_norm': 1, 'virtual_batch_size': 500, 'sample_size': 50_000}	):
		## Model
		self.model      = model
		self.model_type = model_type
		self.loss_fn    = loss_fn # F.nll_loss
		# Train parameters
		self.train_params = train_params

		## Optimizer
		self.optimizer_name   = optimizer_name
		self.local_opt_params = optimizer_params
		# Create optimizer
		self.optimizer = self.__generate_opt(self.optimizer_name, **self.local_opt_params)

		## Privacy engine
		self.security = security
		self.security_params = security_params
		# Generate privacy engine
		self.privacy_engine  = self.__generate_privacy_engine(self.security, **self.security_params)
		# Attach it to the optimizer
		if self.privacy_engine:
			self.privacy_engine.attach(self.optimizer)

	def eval(self):
		return self.model.eval()

	def train(self):
		return self.model.train()

	def parameters(self):
		return self.model.parameters()

	def to(self, device='cpu'):
		return self.model.to(device)

	def state_dict(self):
		# State dict of the current weights and biases the the network 
		return self.model.state_dict()

	def load_state_dict(self, model_dict):
		return self.model.load_state_dict(model_dict)

	def __call__(self, data):
		return self.model(data)

	def predict(self, data, prob=False):
		self.eval()
		predict_prob = self.model(data)
		predict = output.argmax(dim=1, keepdim=True)
		if prob == False:
			return predict
		else:
			return predict_prob, predict

	def test(self, data_loader, device='cpu'):
		def accuracy(preds, labels):
			return (preds == labels).mean()

		self.eval()
		self.to(device)
		losses   = []
		top1_acc = []

		with torch.no_grad():
			for data, target in data_loader:
				data, target = data.to(device), target.to(device)
				output = self.model(data)
				loss   = self.loss_fn(output, target).item()
				losses.append(loss)
				preds  = np.argmax(output.detach().cpu().numpy(), axis=1)
				labels = target.detach().cpu().numpy()
				acc = accuracy(preds, labels) # measure accuracy and record loss
				top1_acc.append(acc)

		test_loss = np.mean(losses)
		accuracy  = np.mean(top1_acc) * 100

		return accuracy, test_loss

	def step(self, data_loader, device='cpu'):
		self.train()
		self.to(device)
		epochs = self.train_params['epochs']

		if not self.privacy_engine: # regular pytorch train iteration
			for e in range(epochs):
				for batch_idx, (data, target) in enumerate(data_loader):
					data, target = data.to(device), target.to(device)
					output = self.model(data)
					loss   = self.loss_fn(output, target)
					self.optimizer.zero_grad()
					loss.backward()
					self.optimizer.step()

		else: # use privacy engine
			batch_size         = data_loader.batch_size
			virtual_batch_size = self.privacy_engine.batch_size
			# assert virtual_batch_size % batch_size == 0, 'virtual_batch_size should be divisible by batch_size'
			# virtual_batch_rate = int(virtual_batch_size / batch_size)
			self.privacy_engine.to(device)
			if self.privacy_engine.secure_rng is False:
				self.privacy_engine.random_number_generator = torch.Generator(device)
			for e in range(epochs):
				for i, (data, target) in enumerate(data_loader):
					data, target = data.to(device), target.to(device)
					output = self.model(data)
					loss   = self.loss_fn(output, target)
					loss.backward()
					self.optimizer.step()
					self.optimizer.zero_grad() # not needed when using differential privacy

					## Virtual step not supported yet
					# if ((i + 1) % virtual_batch_rate == 0) or ((i + 1) == len(data_loader)):
					# 	self.optimizer.step()
					# else:
					# 	self.optimizer.virtual_step() # take a virtual step

		return loss.item()


	def __generate_opt(self, name='Adam', **kwargs):
		if name == 'Adam':
			optimizer = torch.optim.Adam(self.model.parameters(), **kwargs)

		elif name == 'SGD':
			optimizer = torch.optim.SGD(self.model.parameters(), **kwargs)

		elif name == 'RMSprop':
			optimizer = torch.optim.RMSprop(self.model.parameters(), **kwargs)

		else:
			raise Exception('Unsupported optimizer name. Supported: Adam, SGD, RMSProp')

		return optimizer

	def __generate_privacy_engine(self, name=None, **kwargs):
		if not name: # Default is None
			privacy_engine = None
		else:
			if name == 'DP':
				privacy_engine = PrivacyEngine(
					self.model,
					batch_size       = kwargs['virtual_batch_size'],
					sample_size      = kwargs['sample_size'],
					alphas           = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
					noise_multiplier = kwargs['noise_multiplier'],
					max_grad_norm    = kwargs['max_grad_norm'],
				)
			else:
				raise Exception('Unsupported privacy engine name. Supported: Differential Privacy (DP)')

		return privacy_engine