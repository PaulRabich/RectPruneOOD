import os

import numpy as np

from rect_prune_ood.lib.util.paths import get_method_result_path


def load_scores_for_method(
	method_name: str = 'baseline',
	score_function: str = 'energy_score',
	base_dataset: str = 'imagenet',
) -> dict:
	"""
	Loads all scores for a specified method and base dataset.

	Args:
	method_name (str): Name of the method.
	score_function (str): Name of the score function.
	base_dataset (str): Name of the base dataset. Mainly used for the path. Defaults to "imagenet".

	Returns:
	dict: Dictionary containing the scores for the specified method for all datasets.
	"""

	method_result_path = get_method_result_path(
		base_dataset=base_dataset, method_name=method_name
	)

	scores = {}

	# Loop over all files in the method result directory
	for file_name in os.listdir(method_result_path):
		if 'activations' in file_name:  # skip activation files
			continue
		scores[file_name[:-4]] = np.load(
			os.path.join(method_result_path, file_name), allow_pickle=True
		).item()[score_function]

	return scores
