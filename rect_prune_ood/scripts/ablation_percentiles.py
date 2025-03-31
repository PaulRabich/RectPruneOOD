import argparse
import csv
import os
import subprocess

import numpy as np
from tqdm import tqdm

from rect_prune_ood.lib.eval.metrics import cal_metric, cal_metric_near_ood
from rect_prune_ood.lib.experiments.run_experiment import run_single_experiment
from rect_prune_ood.lib.util.paths import get_top_level_path

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


def run_mask_creation(
	backbone: str, percentile: int, name: str, severity: int, value: int
):
	command = [
		'python',
		'./create_mask.py',
		'--name',
		name,
		'--backbone',
		backbone,
		'--percentile',
		str(percentile),
		'--severity',
		str(severity),
		'--values',
		str(value),
	]

	process = subprocess.Popen(
		command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
	)

	# Stream output line-by-line
	for line in process.stdout:
		print(line, end='')  # Print as it comes

	process.wait()  # Ensure process completes


def run_single_experiment_cmd(backbone: str, dataset: str, method_name: str):
	command = [
		'python',
		'./calculate_score.py',
		'--method_names',
		method_name,
		'--datasets',
		'imagenet',
		dataset,
		'--backbone',
		backbone,
		'--save_results',
		'False',
	]

	process = subprocess.Popen(
		command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
	)

	# Stream output line-by-line
	for line in process.stdout:
		print(line, end='')  # Print as it comes

	process.wait()  # Ensure process completes


def validate_percentiles(percentiles):
	for percentile in percentiles:
		if not (0 <= percentile <= 100):
			raise argparse.ArgumentTypeError(
				f'Percentile {percentile} is not between 0 and 100.'
			)
	return percentiles


def parse_args():
	parser = argparse.ArgumentParser(
		description='CLI that allows to recreate the ablation study over pruning percentiles for the report.'
	)

	parser.add_argument(
		'--severity',
		type=int,
		default=1,
		help='Imagenet-C severity level (default: 1).',
	)

	parser.add_argument(
		'--backbone',
		type=str,
		default='resnet50',
		choices=['resnet50', 'mobilenetv2'],
		help='Backbone model name (default: resnet50). Options: [resnet50, mobilenetv2]',
	)

	parser.add_argument(
		'--percentiles',
		type=int,
		nargs='*',
		required=True,
		help='List of integers between 0 and 100. Percentile p corresponds to pruning 100-p%% of the activations.',
	)

	parser.add_argument(
		'--dataset',
		type=str,
		default='openimage-o',
		help='Dataset to run the ablation on (default: openimage-o).',
	)

	return parser.parse_args()


def main():
	args = parse_args()
	percentiles = validate_percentiles(args.percentiles)
	severity = args.severity
	backbone = args.backbone
	dataset = args.dataset

	results_dir = os.path.join(get_top_level_path(), 'results')

	masks_dir = os.path.join(results_dir, 'masks')
	os.makedirs(masks_dir, exist_ok=True)

	for percentile in tqdm(percentiles):
		all_metrics = []

		for value in tqdm(range(1, 20)):
			perturbation_name_ = imagenet_c_perturbations[value - 1]
			perturbation_name = perturbation_name_.replace('_', '')

			mask_name = f'{backbone}-sev{severity}-{perturbation_name}-p{percentile}'

			# Check if the mask file exists
			mask_file_path = os.path.join(masks_dir, f'{mask_name}.npy')

			if not os.path.isfile(mask_file_path):
				print()
				print(
					f'Mask file for {perturbation_name_} does not exist. Running mask creation...'
				)
				print()
				run_mask_creation(
					backbone=backbone,
					percentile=percentile,
					name=mask_name,
					severity=severity,
					value=value,
				)

			method_name = f'apes@{mask_name}'

			run_single_experiment_cmd(
				backbone=backbone,
				dataset=dataset,
				method_name=method_name,
			)

			scores_path = os.path.join(
				results_dir, 'imagenet', backbone, method_name, 'scores'
			)

			# Load imagenet scores
			id_scores = np.load(
				os.path.join(
					scores_path,
					'imagenet.npy',
				),
				allow_pickle=True,
			).item()['energy_score']

			ood_scores = np.load(
				os.path.join(
					scores_path,
					f'{dataset}.npy',
				),
				allow_pickle=True,
			).item()['energy_score']

			# Calculate metrics
			if dataset in ['ninco', 'ssb-hard']:
				metrics = cal_metric_near_ood(id_scores, ood_scores)
			else:
				metrics = cal_metric(id_scores, ood_scores)

			metrics['perturbation'] = perturbation_name_
			all_metrics.append(metrics)

		ablation_results_dir = os.path.join(
			results_dir, 'ablation', dataset, '1_percentiles'
		)

		os.makedirs(ablation_results_dir, exist_ok=True)

		csv_file = os.path.join(
			ablation_results_dir,
			f'{backbone}_sev_{severity}_p{percentile}.csv',
		)

		os.makedirs(os.path.dirname(csv_file), exist_ok=True)

		with open(csv_file, mode='w') as file:
			# ONLY write perturbation name, FPR and AUROC
			writer = csv.DictWriter(file, fieldnames=['perturbation', 'FPR', 'AUROC'])
			writer.writeheader()
			for metric in all_metrics:
				writer.writerow(
					{
						'perturbation': metric['perturbation'],
						'FPR': metric['FPR'],
						'AUROC': metric['AUROC'],
					}
				)


if __name__ == '__main__':
	main()
