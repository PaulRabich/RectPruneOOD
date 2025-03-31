import argparse
import subprocess
import csv
import os

from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from lightning import Trainer
from torch import Tensor
from torch.nn import Module

from rect_prune_ood.lib.eval.metrics import cal_metric
from rect_prune_ood.lib.experiments.backbones import get_resnet50
from rect_prune_ood.lib.experiments.feature_extractor import FeatureExtractor
from rect_prune_ood.lib.experiments.pipeline import Pipeline
from rect_prune_ood.lib.util.data_loader import get_data_loader
from rect_prune_ood.lib.util.paths import get_top_level_path

SEVERITY = 1

ablation_1_results = os.path.join(
	get_top_level_path(), 'results', 'ablation', '1_percentiles'
)

results_dir = os.path.join(get_top_level_path(), 'results', 'ablation', '2_perturbation_combination')
imagenet_c_results_dir = os.path.join(results_dir, 'imagenet', 'baseline', 'imagenet_c')
# data_dir = os.path.join(get_top_level_path(), 'results', f'percentile_mega_ablation_sev_{SEVERITY}')
data_dir = os.path.join(
	get_top_level_path(), 'results', f'ablation_1_ashs_apes_sev_{SEVERITY}'
)

print('Loading ImageNet-C activations')

imagenet_c_activations = np.load(
	os.path.join(
		imagenet_c_results_dir, f'imagenet_c_activations_severity_{SEVERITY}.npy'
	)
)

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

for percentile in [95]:  # [85, 90, 95]:
	all_metrics = []

	file_path = os.path.join(data_dir, f'percentile_{percentile}.csv')
	if os.path.exists(file_path):
		df = pd.read_csv(file_path)

		df_sorted = df.sort_values(by=['AUROC', 'FPR'], ascending=[False, True])
		perturbation_order = df_sorted['perturbation'].values.tolist()

	print('#' * 50)
	print(f'Percentile: {percentile}')
	print(f'Perturbation Order: {perturbation_order}')
	print('#' * 50)

	for perturbation in range(len(imagenet_c_perturbations)):
		print()
		print('#' * 50)
		print(f'Percentile: {percentile}')
		print(f'Num Perturbations: {perturbation + 1}')
		print('#' * 50)

		perturbation_index = []
		for j in range(perturbation + 1):
			perturbation_index.append(
				imagenet_c_perturbations.index(perturbation_order[j])
			)

		activations = np.concatenate(
			[imagenet_c_activations[j] for j in perturbation_index], axis=1
		)

		variance = np.var(activations, axis=1)
		variance = torch.tensor(variance)

		p = np.percentile(variance, percentile)

		mask = torch.where(variance <= p, 1, 0)
		mask = mask.view(variance.shape[0], 1, 1).cuda()

		#################
		### ImageNet ####
		#################

		feature_extractor = FeatureExtractor(get_resnet50(), 'avgpool')

		fc_layer = feature_extractor.get_fc_layer()

		pipeline = Pipeline(
			feature_extractor, 'ashs', fc_layer, caputure_activations=False
		)
		pipeline.method_list.append(APESP(mask))

		dataloader, _ = get_data_loader('imagenet', batch_size=16)

		# Trainer
		trainer = Trainer(logger=None)
		trainer.test(pipeline, dataloader)

		results = pipeline.results

		imagenet_scores = results['energy_score']

		####################
		### OpenImage-O ####
		####################

		feature_extractor = FeatureExtractor(get_resnet50(), 'avgpool')

		fc_layer = feature_extractor.get_fc_layer()

		pipeline = Pipeline(
			feature_extractor, 'ashs', fc_layer, caputure_activations=False
		)
		pipeline.method_list.append(APESP(mask))

		dataloader, _ = get_data_loader('openimage-o', batch_size=16)

		# Trainer
		trainer = Trainer(logger=None)
		trainer.test(pipeline, dataloader)

		results = pipeline.results

		openimage_o_scores = results['energy_score']

		###############
		### Metrics ###
		###############

		metrics = cal_metric(imagenet_scores, openimage_o_scores)
		metrics['num_perturbations'] = perturbation + 1

		all_metrics.append(metrics)

	csv_file = os.path.join(
		results_dir,
		f'ablation_2_ashs_apes_sev_{SEVERITY}',
		f'percentile_{percentile}.csv',
	)

	os.makedirs(os.path.dirname(csv_file), exist_ok=True)

	with open(csv_file, mode='w') as file:
		writer = csv.DictWriter(file, fieldnames=all_metrics[0].keys())
		writer.writeheader()
		for metric in all_metrics:
			writer.writerow(metric)

def run_mask_creation(backbone: str, percentile: int, name: str, severity: int, value: int):
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

	process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

	# Stream output line-by-line
	for line in process.stdout:
		print(line, end='')  # Print as it comes

	process.wait()  # Ensure process completes

def run_single_experiment(backbone: str, dataset: str, method_name: str):
	command = [
		'python',
		'./calculate_score.py',
		'--method_names',
		method_name,
		'--datasets',
		'imagenet',
		dataset,
		'--backbone',
		backbone
	]
	
	process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

	# Stream output line-by-line
	for line in process.stdout:
		print(line, end='')  # Print as it comes

	process.wait()  # Ensure process completes

def validate_percentiles(percentiles):
	for percentile in percentiles:
		if not (0 <= percentile <= 100):
			raise argparse.ArgumentTypeError(
				f"Percentile {percentile} is not between 0 and 100."
			)
	return percentiles

def parse_args():
	parser = argparse.ArgumentParser(
		description='CLI that allows to recreate the ablation study over perturbation combinations for the report.'
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

	for percentile in tqdm(percentiles):

		if not os.path.isfile(
			os.path.join(ablation_1_results, f'{backbone}_sev_{severity}_p{percentile}.csv')
		):
			print(f'Percentile {percentile} not found in ablation 1 results. Skipping...')
			continue