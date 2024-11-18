from typing import Literal
import os

import torch
from torch.nn import Module
import torch.nn.functional as F
import numpy as np

from rect_prune_ood.lib.experiments.feature_extractor import FeatureExtractor
from rect_prune_ood.lib.experiments.pipeline import Pipeline
from rect_prune_ood.lib.experiments.backbones import get_resnet50
from rect_prune_ood.lib.methods.rect_prune_methods import get_classes
from rect_prune_ood.lib.util import get_in_data_loader, get_out_data_loader, get_top_level_path
from rect_prune_ood.lib.eval.scores import get_score
from tqdm import tqdm


prune_types = Literal["", "ashb", "ashp", "ashs", "dice", "line"]
#weight_pruning_types= Literal["Methoden", "hier", "einfügen"]
rectification_types = Literal["", "react", "bats"]
dataset_types = Literal["cifar10"]
backbones = Literal["resnet50"]


def get_backbone(backbone: backbones):
    if backbone == "resnet50":
        return get_resnet50()

def evalualte_performance(pipeline:Module, dataset:str, result_dir:str):

    dataloader, num_classes = get_in_data_loader(dataset, shuffle=False)
    pipeline.to("cuda")

    energy_scores = []
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        image, label = batch

        images = image.cuda()
        labels = label.cuda()

        inputs = images.float()
        logits = pipeline(inputs)

        logits = F.softmax(logits, dim=1)
        logits = logits.detach().cpu()
        
        # compute the metrics
        scores = get_score(inputs=inputs, model=pipeline, method="energy", logits=logits)
        energy_scores.append(scores)


    energy_scores = np.concatenate(energy_scores).flatten()
    os.makedirs(result_dir, exist_ok=True)
    final_path = os.path.join(result_dir,"energy_scores.np") 
    with open(final_path, "w") as f:
        np.savetxt(f, energy_scores)
        


def run_experiment(backbone: backbones, layer:str, dataset:dataset_types, rectification_method:rectification_types, prune_method:prune_types):
    backbone = get_backbone(backbone)
    feature_extractor = FeatureExtractor(backbone, layer)

    #return appropriate methods
    rect_cls, prune_cls= get_classes(rectification=rectification_method, pruning=prune_method)

    rect_method = rect_cls()
    prune_method = prune_cls(fc_layer=feature_extractor.get_fc_layer())
    
    pipeline = Pipeline(feature_extractor, rect_method, prune_method)

    result_dir = os.path.join(get_top_level_path(), "results")

    evalualte_performance(pipeline, dataset, result_dir=result_dir)
    