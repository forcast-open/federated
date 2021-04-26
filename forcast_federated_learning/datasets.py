import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
from sklearn.datasets import load_iris, load_boston

class StructuredDataset(Dataset):
	"""Custom dataset for structured data."""
	
	def __init__(self, X, y, transform=None, categorical=True):
		"""
		Args:
			numpy_array (array): Matrix of scructured features.
			numpy_array (array): List of targets.
			transform (callable, optional): Optional transform to be applied
				on a sample.
			categorical (bool, optional): Set to False if the targets are not categories (e.g. regression) 
		"""
		# Check inputs are 2D and 1D arrays (X and y respectively)
		assert len(X.shape) == 2, 'X value expected two dimensional numpy matrix'
		assert len(y.shape) == 1, 'y value expected one dimensional numpy array'
		# Check same number of datapoints
		assert y.shape[0] == X.shape[0], 'X and y do not have the same number of datapoints'
		# Set attributes
		self.categorical = categorical
		self.data    = Variable(torch.from_numpy(X)).float()
		self.targets = None
		if self.categorical == True: 
			# Only calculate num_calsses if there are categories in the outputs
			self.num_classes = len(np.unique(self.targets))
			# Targets are ints (for nn.CrossEntropyLoss() use for example)
			self.targets = Variable(torch.from_numpy(y)).long()
		else: 
			# Expand y dimension
			y = np.expand_dims(y, axis=1)
			# Targets are floats (for nn.MSELoss() use for example)
			self.targets = Variable(torch.from_numpy(y)).float()			
		self.num_features = self.data.shape[1]
		self.transform = transform

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		
		sample = (self.data[idx], self.targets[idx])

		if self.transform:
			sample = self.transform(sample)

		return sample
	
	def __repr__(self):
		if self.categorical is True:
			message = ['Structured Dataset',
					   '\n\t'+'Number of datapoints: '+str(self.__len__()),
					   '\n\t'+'Number of features: '+str(self.num_features),
					   '\n\t'+'Number of classes: '+str(self.num_classes)]
		else:
			message = ['Structured Dataset',
					   '\n\t'+'Number of datapoints: '+str(self.__len__()),
					   '\n\t'+'Number of features: '+str(self.num_features)]
		return ''.join(message)

class ImageDataset(Dataset):
	"""Custom dataset for structured data."""
	
	def __init__(self, X, y, transform=None, categorical=True):
		"""
		Args:
			numpy_array (array): Matrix of scructured features.
			numpy_array (array): List of targets.
			transform (callable, optional): Optional transform to be applied
				on a sample.
			categorical (bool, optional): Set to False if the targets are not categories (e.g. regression) 
		"""
		# Check inputs are 2D and 1D arrays (X and y respectively)
		assert len(X.shape) in [3, 4], 'X value expected three or four dimensional numpy matrix'
		assert len(y.shape) == 1, 'y value expected one dimensional numpy array'
		# Expand dim if X is 3D
		if len(X.shape) == 3:
			X = np.expand_dims(X,1)
		# Check same number of datapoints
		assert y.shape[0] == X.shape[0], 'X and y do not have the same number of datapoints'
		# Set attributes
		self.categorical = categorical
		self.data    = Variable(torch.from_numpy(X)).float()
		self.targets = None
		if self.categorical == True:
			# Only calculate num_calsses if there are categories in the outputs
			self.num_classes = len(np.unique(self.targets))
			# Targets are ints (for nn.CrossEntropyLoss() use for example)
			self.targets = Variable(torch.from_numpy(y)).long()
		else: 
			# Expand y dimension
			y = np.expand_dims(y, axis=1)
			# Targets are floats (for nn.MSELoss() use for example)
			self.targets = Variable(torch.from_numpy(y)).float()			
		self.shape_images = self.data[0,:].shape
		self.transform = transform

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		
		sample = (self.data[idx,:], self.targets[idx])

		if self.transform:
			sample = self.transform(sample)

		return sample
	
	def __repr__(self):
		if self.categorical is True:
			message = ['Structured Dataset',
					   '\n\t'+'Number of datapoints: '+str(self.__len__()),
					   '\n\t'+'Shape of images: '+str(self.shape_images),
					   '\n\t'+'Number of classes: '+str(self.num_classes)]
		else:
			message = ['Structured Dataset',
					   '\n\t'+'Number of datapoints: '+str(self.__len__()),
					   '\n\t'+'Shape of images: '+str(self.shape_images)]
		return ''.join(message)


#### Load Iris Dataser ####
def load_scikit_iris():
	data_dict = load_iris()
	X = data_dict['data']
	y = data_dict['target']
	target_names  = data_dict['target_names']
	feature_names = data_dict['feature_names']
	df_features = pd.DataFrame(dict(zip(feature_names, X.T)))
	df_target = pd.DataFrame(dict(zip(['target','label'],[map(lambda i: target_names[i], y), y])))
	df_data = pd.concat([df_features, df_target], axis=1)
	return (X, y, df_data, target_names)

#### Load Iris Dataser ####
def load_scikit_boston():
	boston_dataset = load_boston()
	X = boston_dataset['data']
	y = boston_dataset['target']
	s = boston_dataset['DESCR'] # description string
	df_boston = pd.DataFrame(boston_dataset.data,   columns=boston_dataset.feature_names)
	df_target = pd.DataFrame(boston_dataset.target, columns=['MEDV'])
	df_data   = pd.concat([df_boston, df_target], axis=1)
	return (X, y, df_data, s)