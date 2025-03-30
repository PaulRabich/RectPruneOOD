import argparse
import csv
import os
from typing import List

import numpy as np
from tqdm import tqdm

from rect_prune_ood.lib.eval.metrics import cal_metric, cal_metric_near_ood
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
		type=str,
		nargs='+',
		required=True,
		help="A List of dataset names to be used. Examples are 'imagenet', 'inat', 'dtd', 'sun', 'places', 'ninco', 'ssb-hard'.",
	)

	parser.add_argument(
		'--backbone',
		type=str,
		choices=['resnet50', 'mobilenetv2'],
		default='resnet50',
		help='The backbone model to be used. Choose between "resnet50" and "mobilenetv2".',
	)

	args = parser.parse_args()

	return args


def calculate_score(method_names: List[str], datasets: List[str], backbone: str):
	results_dir = os.path.join(get_top_level_path(), 'results')
	metrics_path = os.path.join(results_dir, 'final_metrics.csv')

	# if file does not exist, create it with headers
	if not os.path.isfile(metrics_path):
		with open(metrics_path, 'w', newline='') as f:
			writer = csv.writer(f)
			writer.writerow(
				['backbone', 'method', 'id_dataset', 'ood_dataset', 'FPR', 'AUROC']
			)

	# Create results dir if needed
	os.makedirs(results_dir, exist_ok=True)

	for method_name in tqdm(method_names):
		run_single_experiment(
			backbone=backbone,
			layer='avgpool',
			method_name=method_name,
			capture_activations=False,
			dataset='imagenet',
		)
		for dataset in tqdm(datasets):
			run_single_experiment(
				backbone=backbone,
				layer='avgpool',
				method_name=method_name,
				capture_activations=False,
				dataset=dataset,
			)
			# Load scores
			id_scores = np.load(
				get_method_result_path(backbone=backbone, method_name=method_name)
				+ '/scores/imagenet.npy',
				allow_pickle=True,
			).item()['energy_score']

			ood_scores = np.load(
				os.path.join(
					get_method_result_path(backbone=backbone, method_name=method_name),
					f'scores/{dataset}.npy',
				),
				allow_pickle=True,
			).item()['energy_score']

			# Calculate metrics
			if dataset in ['ninco', 'ssb-hard']:
				metrics = cal_metric_near_ood(id_scores, ood_scores)
			else:
				metrics = cal_metric(id_scores, ood_scores)

			print(
				f'Method: {method_name}, Dataset: {dataset}, FPR: {metrics["FPR"]}, AUROC: {metrics["AUROC"]}'
			)

			# Save metrics to CSV
			with open(metrics_path, 'a', newline='') as f:
				writer = csv.writer(f)
				writer.writerow(
					[
						backbone,
						method_name,
						'imagenet',
						dataset,
						metrics['FPR'],
						metrics['AUROC'],
					]
				)


def main():
	args = parse_args()
	method_names = args.method_names
	datasets = args.datasets
	backbone = args.backbone
	calculate_score(method_names, datasets, backbone)

	print(
		f'Scores calculated and saved to {os.path.join(get_top_level_path(), "results/ood_scores.csv")}'
	)


if __name__ == '__main__':
	main()
