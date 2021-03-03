from opacus import PrivacyEngine as OpacusPrivacyEngine
from opacus.dp_model_inspector import DPModelInspector
from opacus.utils import module_modification

def PrivacyEngine(fed_model, **kwargs):

	# max_grad_norm needs to be a list, if not, code breaks the training models with custom datasets structures
	max_grad_norm  = kwargs['max_grad_norm']
	if not isinstance(max_grad_norm, list):
		max_grad_norm = [max_grad_norm]

	# Secure random number generator
	if 'secure_rng' in list(kwargs.keys()):
		secure_rng = kwargs['secure_rng']
	else: 
		secure_rng = False

	privacy_engine = OpacusPrivacyEngine(
										fed_model.model,
										batch_size       = kwargs['batch_size'],
										sample_size      = kwargs['sample_size'],
										alphas           = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
										noise_multiplier = kwargs['noise_multiplier'],
										max_grad_norm    = max_grad_norm,
										target_delta     = kwargs['target_delta'],
										secure_rng       = secure_rng,
									)
	
	return privacy_engine