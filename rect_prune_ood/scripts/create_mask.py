import argparse
import os

import numpy as np
from tqdm import tqdm

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


def validate_values(values):
	for v in values:
		if v < 1 or v > 19:
			raise argparse.ArgumentTypeError(f'Value {v} is out of range (1–19)')

	duplicates = {v for v in values if values.count(v) > 1}
	if duplicates:
		raise argparse.ArgumentTypeError(
			f'Value list contains duplicates: {sorted(duplicates)}'
		)

	return values


def validate_percentile(percentile):
	if percentile < 0 or percentile > 100:
		raise argparse.ArgumentTypeError(
			f'Percentile {percentile} is out of range (0–100)'
		)
	return percentile


def validate_backbone(backbone):
	if backbone not in ['resnet50', 'mobilenetv2']:
		raise argparse.ArgumentTypeError(
			f'Backbone {backbone} is not supported. Choose from ["resnet50", "mobilenetv2"].'
		)
	return backbone


def parse_args():
	parser = argparse.ArgumentParser(
		description='A CLI that accepts a name and a list of integers (1–19).'
	)

	parser.add_argument(
		'--backbone', type=str, required=True, help='Backbone model name.'
	)
	parser.add_argument(
		'--percentile', type=float, required=True, help='Percentile for mask creation.'
	)
	parser.add_argument('--name', type=str, required=True, help='Your name (string).')
	parser.add_argument(
		'--severity',
		type=int,
		default=1,
		help='Imagenet-C severity level (default: 1).',
	)

	parser.add_argument(
		'--values',
		type=int,
		nargs='*',
		required=True,
		help='List of integers between 1 and 19.',
	)

	args = parser.parse_args()
	if '_' in args.name:
		raise argparse.ArgumentTypeError('Name cannot contain underscores.')
	args.values = validate_values(args.values)
	args.percentile = validate_percentile(args.percentile)
	args.backbone = validate_backbone(args.backbone)

	return args


def main():
	args = parse_args()
	name = args.name
	values = args.values
	backbone = args.backbone
	percentile = args.percentile
	severity = args.severity

	print(f'Creating mask for {name} with values {values} and percentile {percentile}.')

	activations_path = os.path.join(
		get_top_level_path(), 'results', 'imagenet', backbone, 'baseline', 'activations'
	)

	activations = []

	for value in tqdm(values):
		perturbation_name = imagenet_c_perturbations[value - 1]

		# Check if the file exists
		file_path = os.path.join(
			activations_path, f'imagenet-c_sev{severity}_{perturbation_name}.npy'
		)

		if not os.path.isfile(file_path):
			print(
				f'Activation file for {perturbation_name} does not exist. Running experiment...'
			)
			run_single_experiment(
				backbone=backbone,
				layer='avgpool',
				dataset='imagenet-c',
				method_name='baseline',
				capture_activations=True,
				severity=severity,
			)

		activations.append(
			np.load(
				os.path.join(
					activations_path,
					f'imagenet-c_sev{severity}_{perturbation_name}.npy',
				)
			)
		)

	activations = np.concatenate(activations, axis=0)

	variance = np.var(activations, axis=0)

	percentile = np.percentile(variance, percentile)

	mask = np.where(variance <= percentile, 1, 0)

	mask_dir = os.path.join(get_top_level_path(), 'results', 'masks')

	os.makedirs(mask_dir, exist_ok=True)

	mask_path = os.path.join(mask_dir, f'{name}.npy')
	np.save(mask_path, mask)


if __name__ == '__main__':
	main()