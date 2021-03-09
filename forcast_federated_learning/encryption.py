import tenseal as ts
import torch
from collections import OrderedDict
import numpy as np
import copy as cp

class SerializedEncTensor:
	def __init__(self, enc_tensor_bytes, encrypted=False):
		if not isinstance(enc_tensor_bytes, bytes):
			raise TypeError('Invalid input types context: {}, expected bytes'.format(type(enc_tensor_bytes)))
			
		self.data = enc_tensor_bytes
		
	def deserialize(self, context):
		if not isinstance(context, ts.Context):
			raise TypeError('Invalid input types context: {}'.format(type(context)))

		return ts.CKKSTensor.load(context, self.data)
	
class EncStateDict:
	def __init__(self, state_dict, encrypted=False):
		if not isinstance(state_dict, OrderedDict):
			raise TypeError('Invalid input types serialized_state_dict: {}'.format(type(serialized_state_dict)))

		for tensor in state_dict.values():
			if not isinstance(tensor, ts.CKKSTensor if encrypted else torch.Tensor):
				raise TypeError('Invalid input types state_dict values: {}, expected: {}'.format(type(ser_enc_tensor)), ts.CKKSTensor if encrypted else torch.Tensor)
		
		self.state_dict = state_dict
		self.encrypted  = encrypted
	
	def is_encrypted(self):
		return self.encrypted
		
	def __encrypt(self, context, state_dict):
		if not isinstance(context, ts.Context):
			raise TypeError('Invalid input types context: {}'.format(type(context)))
			
		aux_state_dict = state_dict.copy()
		for name, tensor in aux_state_dict.items():
			aux_state_dict[name] = ts.ckks_tensor(context, tensor)
		
		return aux_state_dict
	
	def encrypt(self, context):
		if not isinstance(context, ts.Context):
			raise TypeError('Invalid input types context: {}'.format(type(context)))
			
		if self.encrypted == True:
			raise TypeError('Data is already encrypted')
			
		self.state_dict = self.__encrypt(context, self.state_dict)
		self.encrypted  = True
		
		return self
	
	def decrypt(self, secret_key):
		if not isinstance(secret_key, ts.enc_context.SecretKey):
			raise TypeError('Invalid input types secret_key: {}'.format(type(secret_key)))
		
		if self.encrypted == False:
			raise TypeError('Data is already decrypted.')
		
		state_dict = self.state_dict.copy()
		for name, enc_tensor in self.items():
			shape  = enc_tensor.shape
			tensor = enc_tensor.decrypt(secret_key).raw
			tensor = np.array(tensor).reshape(shape)
			tensor = torch.from_numpy(tensor).float()
			state_dict[name] = tensor
			
		return state_dict
	
	def serialize(self):
		"""
		Serialize encrypted data in the OrderedDict.
		"""
		if self.encrypted == False:
			raise TypeError('Data needs to be encrypted to be serialized. i.e. enc_state_dict.encrypt(context).')
		
		state_dict = self.state_dict.copy()
		for name, enc_tensor in self.items():
			state_dict[name] =  SerializedEncTensor( enc_tensor.serialize() )
			
		return state_dict # serialized state dictionary
	
	@classmethod
	def load(cls, context, serialized_state_dict):
		if not isinstance(context, ts.Context):
			raise TypeError('Invalid input types context: {}'.format(type(context)))

		for ser_enc_tensor in serialized_state_dict.values():
			if not isinstance(ser_enc_tensor, SerializedEncTensor):
				raise TypeError('Invalid input types serialized_state_dict values: {}, expected: SerializedEncTensor'.format(type(ser_enc_tensor)))
		
		state_dict = serialized_state_dict.copy()
		for name, ser_enc_tensor in serialized_state_dict.items():
			state_dict[name] =  ser_enc_tensor.deserialize(context)
		
		return cls(state_dict, encrypted=True)
		
			
	def __add__(self, y): # addition
		if isinstance(y, int):
			y = float(y)

		if isinstance(y, EncStateDict): # 'Operations only supported between EncStateDict instances'
			assert len(self) == len(y), f'Operators have different number of layers: {len(self)}, {len(y)}'
			for (x_name, x_tensor), (y_name, y_tensor) in zip(self.items(), y.items()):
				assert x_name == y_name, f'EncStateDict have different weight names: {x_name}, {y_name}' 
				assert x_tensor.shape == y_tensor.shape, f'Tensors of {name_x} have different shapes: {x_tensor.shape}, {y_tensor.shape}'
			
			z = cp.deepcopy(self)
			for name in self.keys():
				z.state_dict[name] = self.state_dict[name] + y.state_dict[name]
		
		elif isinstance(y, float):
			z = cp.deepcopy(self)
			for name in self.keys():
				z.state_dict[name] = self.state_dict[name] + y
		
		else:
			raise TypeError('Can only add two EncStateDict or one EncStateDict and a float')
		
		return z
	
	def __radd__(self,y): # make the addition operation commutative
		return self.__add__(y)
		
	def __mul__(self, y): # multiplication
		if isinstance(y, int):
			y = float(y)
		
		if isinstance(y, EncStateDict): # 'Operations only supported between EncStateDict instances'
			assert len(self) == len(y), f'Operators have different number of layers: {len(self)}, {len(y)}'
			for (x_name, x_tensor), (y_name, y_tensor) in zip(self.items(), y.items()):
				assert x_name == y_name, f'EncStateDict have different weight names: {x_name}, {y_name}' 
				assert x_tensor.shape == y_tensor.shape, f'Tensors of {name_x} have different shapes: {x_tensor.shape}, {y_tensor.shape}'
			
			z = cp.deepcopy(self)
			for name in self.keys():
				z.state_dict[name] = self.state_dict[name] * y.state_dict[name]
		
		elif isinstance(y, float):
			z = cp.deepcopy(self)
			for name in self.keys():
				z.state_dict[name] = self.state_dict[name] * y
		
		else:
			raise TypeError('Can only multiply two EncStateDict or one EncStateDict and a float')
		
		return z
	
	def __rmul__(self,y): # make the multiplication operation commutative
		return self.__mul__(y)
	
	def __len__(self):
		return len(self.state_dict)
	
	def items(self):
		return self.state_dict.items()
	
	def keys(self):
		return self.state_dict.keys()
	
	def values(self):
		return self.state_dict.values() 
	
	def __getitem__(self, item):
		return self.state_dict[item]
			
	def __repr__(self):
		return repr(self.state_dict)

class SerializedContext:
	def __init__(self, context):
		if not isinstance(context, ts.Context):
			raise TypeError('Invalid input types context: {}, expected ts.Context'.format(type(enc_tensor_bytes)))
			
		self.context_bytes = context.serielize()
		
	def deserialize(self):
		return ts.context_from(self.context_bytes)

def get_context(scheme_type         = ts.SCHEME_TYPE.CKKS,
				poly_modulus_degree = 8192,
				coeff_mod_bit_sizes = [60, 40, 40, 60],
				seed                = None):
	"""
	Setup TenSEAL context
	"""
	np.random.seed(seed=seed)
	context = ts.context(
			ts.SCHEME_TYPE.CKKS,
			poly_modulus_degree=8192,
			coeff_mod_bit_sizes=[60, 40, 40, 60]
		  )
	context.generate_galois_keys()
	context.global_scale = 2**40
	secret_key = context.secret_key()
	context.make_context_public()
	assert context.is_public(), 'Error context need to be public (with no secret key) to be shared.'

	return context, secret_key # return public context and private key

def load_context(serialized_context):
	return ts.Context.load(serialized_context)