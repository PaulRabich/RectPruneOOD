from typing import List

from torch.nn import Module

from rect_prune_ood.lib.methods.apes import APES
from rect_prune_ood.lib.methods.rect_prune_methods import (
	AshBPruningModule,
	AshPPruningModule,
	AshSPruningModule,
	BatsRectModule,
	DicePruningModule,
	ReactRectModule,
	SCALEModule,
)

module_dict = {
	'ashp': AshPPruningModule,
	'ashb': AshBPruningModule,
	'ashs': AshSPruningModule,
	'scale': SCALEModule,
	'dice': DicePruningModule,
	'react': ReactRectModule,
	'bats': BatsRectModule,
	'apes': APES,
}


def get_module_list(method_name: str) -> List[Module]:
	"""Creates a list of modules based on the method name.

	Args:
		method_name (str): The name of the method to create modules for. The method name can be a combination of module names separated by underscores.

	Raises:
		ValueError: If the method name is invalid or if a module string is not recognized.

	Returns:
		List[Module]: A list of instantiated modules corresponding to the method name and given parameters.
	"""	
	if method_name == 'baseline':
		return []

	modules = []
	modules_str = method_name.split('_')

	for module_str in modules_str:
		module_params = module_str.split('@')
		if len(module_params) == 1:
			module = module_params[0]
			modules.append(module_dict[module]())
		elif len(module_params) == 2:
			module = module_params[0]
			param = module_params[1]
			modules.append(module_dict[module](param))
		else:
			raise ValueError(f'Invalid module string: {module_str}')

	return modules
