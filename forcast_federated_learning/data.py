import torch 

#### Random split into n clients
def random_split(dataset, num_clients, device='cpu', seed=None):
	generator = torch.Generator(device=device)
	if seed is not None:
		generator.manual_seed(seed)

	return torch.utils.data.random_split(dataset=dataset, lengths=[int(len(dataset.data) / num_clients) for _ in range(num_clients)], generator=generator)