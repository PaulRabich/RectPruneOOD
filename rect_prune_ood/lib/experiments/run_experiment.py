from typing import Literal

from lightning import Trainer

from rect_prune_ood.lib.experiments.feature_extractor import FeatureExtractor
from rect_prune_ood.lib.experiments.pipeline import Pipeline
from rect_prune_ood.lib.experiments.backbones import get_resnet50



prune_types = Literal["Methoden", "hier", "einfügen"]
weight_pruning_types= Literal["Methoden", "hier", "einfügen"]
rectification_types = Literal["Methoden", "hier", "einfügen"]
dataset_types = Literal["Methoden", "hier", "einfügen"]
backbones = Literal["resnet50"]


def get_backbone(backbone: backbones):
    if backbone == "resnet50":
        return get_resnet50()

def get_prune_method(prune: prune_types):
    pass

def get_weight_pruning_method(weight_pruning: weight_pruning_types):
    pass

def get_rectification_method(rectification: rectification_types):
    pass


def run_experiment(backbone: backbones, layer:str, dataset:dataset_types, rectification:rectification_types, weight_pruning:weight_pruning_types, prune:prune_types):
    backbone = get_backbone(backbone)
    feature_extractor = FeatureExtractor(backbone, layer)
    rect_method = get_rectification_method(rectification)
    prune_method = get_prune_method(prune)
    pipeline = Pipeline(feature_extractor, rect_method, prune_method)
    
    dataloader = ...
    
    module = Trainer(pipeline)
    ood_performance = module.test(dataloaders=dataloader)

    return ood_performance