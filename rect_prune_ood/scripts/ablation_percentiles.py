import sys
import os
import torch
import csv
import numpy as np

from torch.nn import Module
from torch import Tensor

from lightning import Trainer

sys.path.append("./")

from rect_prune_ood.lib.util.paths import get_top_level_path
from rect_prune_ood.lib.experiments.feature_extractor import FeatureExtractor
from rect_prune_ood.lib.experiments.backbones import get_resnet50
from rect_prune_ood.lib.experiments.pipeline import Pipeline
from rect_prune_ood.lib.util.data_loader import get_data_loader
from rect_prune_ood.lib.eval.metrics import *

results_dir = os.path.join(get_top_level_path(), 'results')
imagenet_c_results_dir = os.path.join(results_dir, 'imagenet', "baseline", "imagenet_c")

print("Loading ImageNet-C activations")

severity = 1

imagenet_c_activations = np.load(os.path.join(imagenet_c_results_dir, f"imagenet_c_activations_severity_{severity}.npy"))

imagenet_c_perturbations = [
    "brightness",
    "defocus_blur",
    "fog",
    "gaussian_blur",
    "glass_blur",
    "jpeg_compression",
    "pixelate",
    "shot_noise",
    "spatter",
    "zoom_blur",
    "contrast",
    "elastic_transform",
    "frost",
    "gaussian_noise", 
    "impulse_noise",
    "motion_blur",
    "saturate",
    "snow",
    "speckle_noise"
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

for percentile in [95]:# [i * 5 for i in reversed(range(2, 19))]:
    if percentile == 0:
        continue

    all_metrics = []

    for perturbation in range(len(imagenet_c_perturbations)):
        print()
        print("#" * 50)
        print(f"Percentile: {percentile}")
        print(f"Perturbation: {imagenet_c_perturbations[perturbation]}")
        print("#" * 50)

        activations = imagenet_c_activations[perturbation]

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

        pipeline = Pipeline(feature_extractor, 'ashs', fc_layer, caputure_activations=False)
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

        pipeline = Pipeline(feature_extractor, 'ashs', fc_layer, caputure_activations=False)
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
        metrics['perturbation'] = imagenet_c_perturbations[perturbation]

        all_metrics.append(metrics)

    csv_file = os.path.join(
        results_dir, f"ablation_1_ashs_apes_sev_{severity}", f'percentile_{percentile}.csv'
    )

    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    with open(csv_file, mode='w') as file:
        writer = csv.DictWriter(file, fieldnames=all_metrics[0].keys())
        writer.writeheader()
        for metric in all_metrics:
            writer.writerow(metric)
