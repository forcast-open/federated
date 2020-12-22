import torch 

#### Wraper for pytorch dataloader method in the library
def DataLoader(dataset, batch_size=1, shuffle=False):
	return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)