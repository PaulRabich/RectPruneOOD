import os


def get_top_level_path():
	# get the absolute path of this file
	abs_path = os.path.dirname(os.path.abspath(__file__))
	# get the parent directory
	top_level_path = os.path.abspath(os.path.join(abs_path, '../../../'))

	return top_level_path


def get_method_result_path(
	base_dataset: str = 'imagenet', method_name: str = 'baseline'
):
	result_dir = os.path.join(
		get_top_level_path(), f'results/{base_dataset}/{method_name}'
	)
	os.makedirs(result_dir, exist_ok=True)

	return result_dir
