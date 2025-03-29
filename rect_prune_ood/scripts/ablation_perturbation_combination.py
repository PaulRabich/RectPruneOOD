import csv
import os

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

results_dir = os.path.join(get_top_level_path(), 'results')
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


class APESP(Module):
	def __init__(self, mask) -> None:
		super(APESP, self).__init__()
		self.mask = mask

	def forward(self, x: Tensor) -> Tensor:
		x = self.apes_p(x)
		return x

	def apes_p(self, x):
		assert x.dim() == 4
		_, c, _, _ = x.shape
		assert self.mask.shape[0] == c

		x = x * self.mask

		return x


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
