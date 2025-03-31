import os
from typing import Literal, Union

import numpy as np
from lightning import Trainer

from rect_prune_ood.lib.experiments.backbones import get_mobilenetV2, get_resnet50
from rect_prune_ood.lib.experiments.feature_extractor import FeatureExtractor
from rect_prune_ood.lib.experiments.mobilenet_feature_extractor import (
	MobileNetFeatureExtractor,
)
from rect_prune_ood.lib.experiments.pipeline import Pipeline
from rect_prune_ood.lib.util.data_loader import (
	get_data_loader,
	get_imagenet_c_subset_loader,
	id_datasets,
	ood_datasets,
	perturbed_datasets,
)
from rect_prune_ood.lib.util.paths import get_method_result_path

prune_types = Literal['', 'ashb', 'ashp', 'ashs', 'dice', 'line']
rectification_types = Literal['', 'react', 'bats']
backbones = Literal['resnet50', 'mobilenetv2']


# get feature extractof matching to backbone
def get_feature_extractor(backbone: backbones, layer: str):
	if backbone == 'mobilenetv2':
		return MobileNetFeatureExtractor(get_mobilenetV2())
	else:
		return FeatureExtractor(get_resnet50(), layer)


def run_single_experiment(
	backbone: backbones,
	layer: str,
	dataset: Union[id_datasets, ood_datasets, perturbed_datasets],
	method_name: str,
	capture_activations: bool = False,
	base_dataset: id_datasets = 'imagenet',
	severity: int = 1,
):
	"""
	Runs a single experiment for a specified dataset and method. Saves the results in the corresponding directory.

	Args:
	backbone (backbones): Backbone to use.
	layer (str): Layer to use.
	dataset (Union[id_datasets, ood_datasets, perturbed_datasets]): Dataset for which to run the experiment.
	method_name (str): Name of the method.
	capture_activations (bool): Whether to capture the activations.
	base_dataset (id_datasets): Base dataset. Mainly used for paths. Defaults to "imagenet".
	severity (int): Severity level for the experiment. Defaults to 1.
	"""
	if dataset == 'imagenet-c':
		_run_imagenet_c_experiment(
			backbone=backbone,
			layer=layer,
			method_name=method_name,
			capture_activations=capture_activations,
			severity=severity,
		)
		return

	# Check whether result file already exists
	result_dir = get_method_result_path(base_dataset, backbone, method_name)

	scores_dir = os.path.join(result_dir, 'scores')
	os.makedirs(scores_dir, exist_ok=True)

	if capture_activations:
		activations_dir = os.path.join(result_dir, 'activations')
		os.makedirs(activations_dir, exist_ok=True)

		activations_exist = os.path.isfile(
			os.path.join(activations_dir, f'{dataset}.npy')
		)

	scores_exist = os.path.isfile(os.path.join(scores_dir, f'{dataset}.npy'))

	if scores_exist and not capture_activations:
		print(f'Score results for {dataset} already exist. Skipping...')
		return
	elif scores_exist and capture_activations and activations_exist:
		print(f'Score and activation results for {dataset} already exist. Skipping...')
		return

	print(f'Running experiment for {dataset}...')

	feature_extractor = get_feature_extractor(backbone, layer)

	fc_layer = fc_layer = feature_extractor.get_fc_layer()

	# get the pipeline and the dataloader
	pipeline = Pipeline(
		feature_extractor,
		method_name,
		fc_layer,
		caputure_activations=capture_activations,
	)

	dataloader, _ = get_data_loader(dataset, batch_size=16)

	trainer = Trainer(logger=None)
	trainer.test(pipeline, dataloader)

	np.save(os.path.join(scores_dir, f'{dataset}.npy'), pipeline.results)

	if capture_activations:
		activations_dir = os.path.join(result_dir, 'activations')
		os.makedirs(activations_dir, exist_ok=True)

		np.save(
			os.path.join(activations_dir, f'{dataset}.npy'),
			pipeline.activations,
		)


def _run_imagenet_c_experiment(
	backbone: backbones,
	layer: str,
	method_name: str,
	capture_activations: bool = False,
	severity: int = 1,
):
	"""Runs the experiment for the ImageNet-C dataset with the specified method.

	Args:
		backbone (backbones): The backbone model to use.
		layer (str): The layer to use for feature extraction.
		method_name (str): The name of the method to apply.
		capture_activations (bool, optional): Whether to caputure and save the activations. Defaults to False.
		severity (int, optional): The severity of the perturbation. Defaults to 1.
	"""
	imagenet_c_pertubations = [
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

	feature_extractor = get_feature_extractor(backbone, layer)

	fc_layer = feature_extractor.get_fc_layer()

	pipeline = Pipeline(
		feature_extractor,
		method_name,
		fc_layer,
		caputure_activations=capture_activations,
	)

	result_dir = get_method_result_path('imagenet', backbone, method_name)
	os.makedirs(result_dir, exist_ok=True)

	for perturbation in imagenet_c_pertubations:
		# Check whether result file already exists
		scores_dir = os.path.join(result_dir, 'scores')
		os.makedirs(scores_dir, exist_ok=True)
		activations_dir = os.path.join(result_dir, 'activations')
		os.makedirs(activations_dir, exist_ok=True)
		scores_exist = os.path.isfile(
			os.path.join(scores_dir, f'imagenet-c_sev{severity}_{perturbation}.npy')
		)
		activations_exist = os.path.isfile(
			os.path.join(activations_dir, f'imagenet-c_sev{severity}_{perturbation}.npy')
		)

		if scores_exist and not capture_activations:
			print(
				f'Score results for imagenet-c {perturbation} already exist. Skipping...'
			)
			continue
		elif (
			scores_exist and capture_activations and activations_exist
		):
			print(
				f'Score and activation results for imagenet-c {perturbation} already exist. Skipping...'
			)
			continue
		print(f'Running experiment for imagenet-c {perturbation}...')

		dataloader = get_imagenet_c_subset_loader(perturbation, severity, batch_size=16)

		# Trainer
		trainer = Trainer(logger=None)
		trainer.test(pipeline, dataloader)

		scores_dir = os.path.join(result_dir, 'scores')
		os.makedirs(scores_dir, exist_ok=True)

		np.save(
			os.path.join(scores_dir, f'imagenet-c_sev{severity}_{perturbation}.npy'),
			pipeline.results,
		)

		if capture_activations:
			activations_dir = os.path.join(result_dir, 'activations')
			os.makedirs(activations_dir, exist_ok=True)

			np.save(
				os.path.join(
					activations_dir, f'imagenet-c_sev{severity}_{perturbation}.npy'
				),
				pipeline.activations,
			)
