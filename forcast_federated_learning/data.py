import torch 
import numpy as np
from forcast_federated_learning.datasets import StructuredDataset

#### Random split into n clients
def random_split(dataset, num_clients, device='cpu', seed=None):
	generator = torch.Generator(device=device)
	if seed is not None:
		generator.manual_seed(seed)

	return torch.utils.data.random_split(dataset=dataset, lengths=[int(len(dataset.data) / num_clients) for _ in range(num_clients)], generator=generator)

def random_clients_data(data_len, num_clients, seed=0):
	'''
	This function creates a random distribution 
	for the clients, i.e. number of images each client 
	has.

	Args:
		data_len: size of the data
		num_clients: number of clients
		seed: seed for random functions
	Outputs:
		rand_dist
	'''
	np.random.seed(seed)
	rand_dist = np.random.randint(low=1, high=2*(data_len/num_clients), size=num_clients)
	rand_dist, sum(rand_dist)
	minidx = np.argmin(rand_dist)
	rand_dist[minidx] += data_len - sum(rand_dist)
	assert sum(rand_dist) == data_len

	return rand_dist

def random_non_iid_split(dataset, num_clients=10, classes_per_client=2, shuffle=True, verbose=False, seed=0):
	'''
	Splits (data, labels) among 'num_clients s.t. every client can holds 'classes_per_client' number of classes

	Args:
		data : [n_data x shape]
		labels : [n_data (x 1)] from 0 to n_labels
		num_clients : number of clients
		classes_per_client : number of classes per client
		shuffle : True/False => True for shuffling the dataset, False otherwise
		verbose : True/False => True for printing some info, False otherwise
		seed: Seed for random functions
	Outputs:
		clients_split : client data into desired format
	'''

	#### dataset ####
	data   = np.array(dataset.data)
	labels = np.array(dataset.targets)

	#### constants #### 
	n_data = data.shape[0]
	n_labels = len(np.unique(labels))

	### client distribution ####
	data_per_client = random_clients_data(len(data), num_clients, seed=seed)
	data_per_client_per_class = [np.maximum(1,nd // classes_per_client) for nd in data_per_client]

	# sort for labels
	data_idcs = [[] for i in range(n_labels)]
	for j, label in enumerate(labels):
		data_idcs[label] += [j]
	if shuffle:
		for i, idcs in enumerate(data_idcs):
			np.random.seed(seed + i)
			np.random.shuffle(idcs)

	# split data among clients
	clients_split = []
	c = 0
	for i in range(num_clients):
		client_idcs = []

		budget = data_per_client[i]
		np.random.seed(seed + i)
		c = np.random.randint(n_labels)
		while budget > 0:
			take = min(data_per_client_per_class[i], len(data_idcs[c]), budget)

			client_idcs += data_idcs[c][:take]
			data_idcs[c] = data_idcs[c][take:]

			budget -= take
			c = (c + 1) % n_labels

		clients_split += [(data[client_idcs], labels[client_idcs])]

	def print_split(clients_split): 
		print("Data split:")
		for i, client in enumerate(clients_split):
			split = np.sum(client[1].reshape(1,-1)==np.arange(n_labels).reshape(-1,1), axis=1)
			print(" - Client {}: {}".format(i,split))
		print()

	if verbose:
		print_split(clients_split)

	#clients_split = np.array(clients_split)
	clients_split = [StructuredDataset(data,label) for data,label in clients_split]

	return clients_split