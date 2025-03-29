import sys
import os
import torch
import csv
import numpy as np
import pandas as pd

from torch.nn import Module
from torch import Tensor
from tqdm import tqdm

from lightning import Trainer

sys.path.append("./")

from rect_prune_ood.lib.util.paths import get_top_level_path
from rect_prune_ood.lib.experiments.feature_extractor import FeatureExtractor
from rect_prune_ood.lib.experiments.backbones import get_resnet50, get_mobilenetV2
from rect_prune_ood.lib.experiments.pipeline import Pipeline
from rect_prune_ood.lib.util.data_loader import get_data_loader
from rect_prune_ood.lib.eval.metrics import *

SEVERITY = 1

results_dir = os.path.join(get_top_level_path(), 'results')
imagenet_c_results_dir = os.path.join(results_dir, 'imagenet', "baseline", "imagenet_c")
data_dir = os.path.join(get_top_level_path(), 'results', f'ablation_1_apes_sev_{SEVERITY}')
#data_dir = os.path.join(get_top_level_path(), 'results', f'ablation_1_ashs_apes_sev_{SEVERITY}')

print("Loading ImageNet-C activations")

base_method = 'scale'
backbone = 'resnet50'

imagenet_c_activations = np.load(os.path.join(imagenet_c_results_dir, f"imagenet_c_activations_severity_{SEVERITY}.npy"))

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
    
percentile = 95
top_p = 19

file_path = os.path.join(data_dir, f'percentile_{percentile}.csv')
if os.path.exists(file_path):
    df = pd.read_csv(file_path)

    df_sorted = df.sort_values(by=["AUROC", "FPR"], ascending=[False, True])
    perturbations = df_sorted["perturbation"].values.tolist()  

perturbations = perturbations[:top_p]

#print(perturbations)

#perturbations = ["gaussian_noise","brightness","saturate","pixelate"]
# perturbations = [
#     "fog",
#     "zoom_blur",
#     "defocus_blur",
#     "glass_blur"
# ]
# perturbation_index = []
# for j in range(top_p):
#     perturbation_index.append(imagenet_c_perturbations.index(perturbation_order[j]))

# activations = np.concatenate(
#     [imagenet_c_activations[j] for j in perturbation_index], 
#     axis=1
# )
activations = np.concatenate(
    [imagenet_c_activations[imagenet_c_perturbations.index(j)] for j in perturbations], 
    axis=1
)

variance = np.var(activations, axis=1)
#variance = np.load(os.path.join(get_top_level_path(), "cache", "imagenet_variance.npy"), allow_pickle="True")
variance = torch.tensor(variance)

p = np.percentile(variance, percentile)

mask = torch.where(variance <= p, 1, 0)
mask = mask.view(variance.shape[0], 1, 1).cuda()


#################
### ImageNet ####
#################

feature_extractor = FeatureExtractor(get_resnet50(), 'avgpool') if backbone == 'resnet50' else FeatureExtractor(get_mobilenetV2(), 'avgpool')

fc_layer = feature_extractor.get_fc_layer()

# pipeline = Pipeline(feature_extractor, 'baseline', fc_layer, caputure_activations=False)
pipeline = Pipeline(feature_extractor, base_method, fc_layer, caputure_activations=False)
# pipeline = Pipeline(feature_extractor, 'scale', fc_layer, caputure_activations=False)
pipeline.method_list.append(APESP(mask))

dataloader, _ = get_data_loader('imagenet', batch_size=16)

# Trainer
trainer = Trainer(logger=None)
trainer.test(pipeline, dataloader)

results = pipeline.results

imagenet_scores = results['energy_score']

####################
### OOD ####
####################

ood_datasets = ['inat', 'sun', 'places', 'dtd', 'ninco', 'ssb_hard']

all_metrics = {}

for ood_dataset in ood_datasets:

    feature_extractor = FeatureExtractor(get_resnet50(), 'avgpool') if backbone == 'resnet50' else FeatureExtractor(get_mobilenetV2(), 'avgpool')

    fc_layer = feature_extractor.get_fc_layer()

    # pipeline = Pipeline(feature_extractor, 'baseline', fc_layer, caputure_activations=False)
    pipeline = Pipeline(feature_extractor, base_method, fc_layer, caputure_activations=False)
    # pipeline = Pipeline(feature_extractor, 'scale', fc_layer, caputure_activations=False)
    pipeline.method_list.append(APESP(mask))

    dataloader, _ = get_data_loader(ood_dataset, batch_size=16)

    # Trainer
    trainer = Trainer(logger=None)
    trainer.test(pipeline, dataloader)

    results = pipeline.results

    ood_scores = results['energy_score']

    ###############
    ### Metrics ###
    ###############

    metrics = cal_metric(imagenet_scores, ood_scores)

    # if ood_dataset in ['ninco', 'ssb_hard']:
    #     alt_metrics = cal_metric(ood_scores, imagenet_scores)
    #     metrics['FPR'] = alt_metrics['FPR']

    all_metrics[ood_dataset] = metrics

print(all_metrics)