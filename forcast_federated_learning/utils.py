import torch 

#### Wraper for pytorch dataloader method in the library
def DataLoader(dataset, batch_size=1, shuffle=False, seed=None):
	if seed: torch.manual_seed(seed)
	return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)