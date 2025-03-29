import os

import numpy as np
import torch
from lightning import Trainer
from torch import Tensor
from torch.nn import Module

from rect_prune_ood.lib.experiments.backbones import get_resnet50
from rect_prune_ood.lib.experiments.feature_extractor import FeatureExtractor
from rect_prune_ood.lib.experiments.pipeline import Pipeline
from rect_prune_ood.lib.util.data_loader import get_data_loader

# module_path = os.path.abspath(os.path.join('..', ''))
# if module_path not in sys.path:
#     sys.path.append(module_path)
# import sys
# sys.path.append("./")
from rect_prune_ood.lib.util.paths import get_top_level_path

results_dir = os.path.join(get_top_level_path(), 'results')
imagenet_c_results_dir = os.path.join(results_dir, 'imagenet', 'baseline', 'imagenet_c')
ablation_study_results_dir = os.path.join(results_dir, 'ablation_study_pertubation')
os.makedirs(ablation_study_results_dir, exist_ok=True)


class APESP(Module):
	def __init__(self, mask) -> None:
		# super(LayerScaler, self).__init__()
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


imagenet_c_pertubations = [
	'gaussian_noise',
	'saturate',
	'brightness',
	'pixelate',
	'snow',
	'motion_blur',
	'elastic_transform',
	'fog',
	'zoom_blur',
	'shot_noise',
	'contrast',
	'defocus_blur',
	'frost',
	'impulse_noise',
	'jpeg_compression',
	'glass_blur',
]

imagenet_c_pertubations_file = [
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


imagenet_c_activations = np.load(
	os.path.join(imagenet_c_results_dir, 'imagenet_c_activations_severity_5.npy')
)
imagenet_activations = np.load(
	os.path.join(results_dir, 'imagenet', 'baseline', 'imagenet_activations.npy')
)
imagenet_activations = np.transpose(imagenet_activations)

for i in range(len(imagenet_c_pertubations)):
	pertubation_index = []
	for j in range(i + 1):
		pertubation_index.append(
			imagenet_c_pertubations_file.index(imagenet_c_pertubations[j])
		)

	# concatenate the activations. data has shape (19, 2048, 50000), goal is to get (2048, x)
	activations = np.concatenate(
		[imagenet_c_activations[j] for j in pertubation_index], axis=1
	)
	# concatenate with imagenet activations
	activations = np.concatenate([activations, imagenet_activations], axis=1)

	variance = np.var(activations, axis=1)
	percentile = np.percentile(variance, 95)

	# Get the indices of values below the 95th percentile
	indices = np.where(variance <= percentile)[0]

	# Construct the mask
	mask = torch.zeros(variance.shape[0])
	mask[indices] = 1
	mask = mask.view(variance.shape[0], 1, 1)
	mask = mask.cuda()

	feature_extractor = FeatureExtractor(get_resnet50(), 'avgpool')

	fc_layer = fc_layer = feature_extractor.get_fc_layer()

	pipeline = Pipeline(
		feature_extractor, 'baseline', fc_layer, caputure_activations=False
	)
	pipeline.method_list = [APESP(mask)]

	dataloader, _ = get_data_loader('imagenet', batch_size=16)

	# Trainer
	trainer = Trainer(logger=None)
	trainer.test(pipeline, dataloader)

	results = pipeline.results

	np.save(
		os.path.join(
			ablation_study_results_dir, f'ablation_study_pertubations_{i + 1}.npy'
		),
		results,
	)
