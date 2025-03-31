import argparse
import os
import csv
from typing import List

import numpy as np
import pandas as pd

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
		help="A List of dataset names to be used. Examples are 'imagenet', 'inat', 'dtd', 'sun', 'places', 'ninco', 'ssb_hard'.",
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

def get_full_spec_scores(method_name: str, backbone: str):
	base_dataset = 'imagenet'

	imagenet_c_perturbations = [
		'brightness',
		'defocus_blur',
		'fog',
		'gaussian_blur',
		'glass_blur',
		'jpeg_compression',
		'pixelate',
		'shot_noise',
		'spatter',
		'zoom_blur',
		'contrast',
		'elastic_transform',
		'frost',
		'gaussian_noise',
		'impulse_noise',
		'motion_blur',
		'saturate',
		'snow',
		'speckle_noise',
	]

	result_dir = os.path.join(
		get_method_result_path(base_dataset=base_dataset, backbone=backbone, method_name=method_name),
		'results',
	)

	all_perturbed_scores = []

	# get all perturbed scores and labels
	for perturbation in imagenet_c_perturbations:
		# print(np.load(result_dir + "/scores/imagenet-c_sev1_" + perturbation + ".npy", allow_pickle=True).item()["energy_score"].shape)
		all_perturbed_scores.append(
			np.load(
				result_dir + '/scores/imagenet-c_sev1_' + perturbation + '.npy',
				allow_pickle=True,
			).item()['energy_score']
		)

	# also get normal ID scores
	all_perturbed_scores.append(
		np.load(result_dir + f'/scores/{base_dataset}.npy', allow_pickle=True).item()[
			'energy_score'
		]
	)

	return np.concatenate(all_perturbed_scores, axis=0)

def calculate_score(method_names: List[str], datasets: List[str], backbone: str):
	results_dir = os.path.join(get_top_level_path(), 'results')
	metrics_path = os.path.join(results_dir, 'final_metrics.csv')

	if not os.path.isfile(metrics_path):
		with open(metrics_path, 'w', newline='') as f:
			writer = csv.writer(f)
			writer.writerow(
				['backbone', 'method', 'id_dataset', 'ood_dataset', 'FPR', 'AUROC']
			)

	for method_name in method_names:
		run_single_experiment(
			backbone=backbone,
			layer='avgpool',
			method_name=method_name,
			capture_activations=False,
			dataset='imagenet',
		)
		run_single_experiment(
			backbone=backbone,
			layer='avgpool',
			method_name=method_name,
			capture_activations=False,
			dataset='imagenet-c',
		)
		for dataset in datasets:
			run_single_experiment(
				backbone=backbone,
				layer='avgpool',
				method_name=method_name,
				capture_activations=False,
				dataset=dataset,
			)

			# Load scores
			id_scores = get_full_spec_scores(method_name, backbone)

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
						'imagenet_full_spec',
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
		f'Scores calculated and saved to {os.path.join(get_top_level_path(), "results/final_metrics.csv")}'
	)


if __name__ == '__main__':
	main()
