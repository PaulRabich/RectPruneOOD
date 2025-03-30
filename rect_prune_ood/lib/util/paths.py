import os
from typing import Literal

backbones = Literal['resnet50', 'mobilenetV2']


def get_top_level_path() -> os.PathLike:
	"""Returns the top level path of the project.

	Returns:
		os.PathLike: The absolute path of the top level directory.
	"""	
	# get the absolute path of this file
	abs_path = os.path.dirname(os.path.abspath(__file__))
	# get the parent directory
	top_level_path = os.path.abspath(os.path.join(abs_path, '../../../'))

	return top_level_path


def get_method_result_path(
	base_dataset: str = 'imagenet',
	backbone: backbones = 'resnet50',
	method_name: str = 'baseline',
) -> os.PathLike:
	"""Returns the path to the results directory for a specific method.

	Args:
		base_dataset (str, optional): Name of the base dataset used. Defaults to 'imagenet'.
		backbone (backbones, optional): Name of the backbone used. Defaults to 'resnet50'.
		method_name (str, optional): Method name used. Defaults to 'baseline'.

	Returns:
		os.PathLike: The path to the results directory for the specified method.
	"""	
	result_dir = os.path.join(
		get_top_level_path(), f'results/{base_dataset}/{backbone}/{method_name}'
	)
	os.makedirs(result_dir, exist_ok=True)

	return result_dir
