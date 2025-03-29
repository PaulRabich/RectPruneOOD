import argparse
import os
from typing import List

import numpy as np
import pandas as pd

from rect_prune_ood.lib.eval.metrics import cal_metric
from rect_prune_ood.lib.experiments.run_experiment import run_single_experiment
from rect_prune_ood.lib.util.paths import get_method_result_path, get_top_level_path


def parse_args():
	parser = argparse.ArgumentParser(
		description='A CLI that accepts a name and a list of integers (1–19).'
	)

	parser.add_argument(
		'--method_names',
		type=str,
		nargs='+',
		required=True,
		help="The names of the methods to be used. Examples are 'react', 'ashs', 'ashp', 'apes@95', etc.",
	)

	parser.add_argument(
		'--datasets',
		type=int,
		nargs='+',
		required=True,
		help="A List of dataset names to be used. Examples are 'imagenet', 'inat', 'dtd', 'sun', 'places', 'ninco', 'ssb_hard'.",
	)

	args = parser.parse_args()

	return args


def calculate_score(method_names: List[str], datasets: List[str]):
	results_dir = os.path.join(get_top_level_path(), 'results')
	metrics_path = os.path.join(results_dir, 'ood_scores.csv')

	# Create results dir if needed
	os.makedirs(results_dir, exist_ok=True)

	for method_name in method_names:
		run_single_experiment(
			backbone='resnet50',
			layer='avgpool',
			method_name=method_name,
			capture_activations=False,
			dataset='imagenet',
		)
		for dataset in datasets:
			run_single_experiment(
				backbone='resnet50',
				layer='avgpool',
				method_name=method_name,
				capture_activations=False,
				dataset=dataset,
			)
			# Load scores
			id_scores = np.load(
				get_method_result_path(backbone='resnet50', method_name=method_name)
				+ '/scores/imagenet.npy',
				allow_pickle=True,
			).item()['energy_score']

			ood_scores = np.load(
				os.path.join(
					get_method_result_path(
						backbone='resnet50', method_name=method_name
					),
					f'scores/{dataset}.npy',
				),
				allow_pickle=True,
			).item()['energy_score']

			# Calculate metrics
			metrics = cal_metric(id_scores, ood_scores)

			# Load existing DataFrame to determine next index
			existing_df = pd.read_csv(metrics_path, index_col='index')
			next_index = len(existing_df)

			result_dict = {
				'index': next_index,
				'method': method_name,
				'ood_dataset': dataset,
				'FPR': metrics['FPR'],
				'AUROC': metrics['AUROC'],
			}

			# Append result and save
			result_df = pd.DataFrame([result_dict]).set_index('index')
			result_df.to_csv(metrics_path, mode='a', header=False)


def main():
	args = parse_args()
	method_names = args.method_names
	datasets = args.datasets
	calculate_score(method_names, datasets)

	print(
		f'Scores calculated and saved to {os.path.join(get_top_level_path(), "results/ood_scores.csv")}'
	)


if __name__ == '__main__':
	main()
