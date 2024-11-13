from typing import Literal

from lightning import Trainer

from rect_prune_ood.lib.experiments.feature_extractor import FeatureExtractor
from rect_prune_ood.lib.experiments.pipeline import Pipeline
from rect_prune_ood.lib.experiments.backbones import get_resnet50
from rect_prune_ood.lib.experiments.rect_prune_methods import get_methods

from rect_prune_ood.lib.util.data_loader import get_in_data_loader, get_out_data_loader


prune_types = Literal["", "ash", "dice", "line"]
#weight_pruning_types= Literal["Methoden", "hier", "einfügen"]
rectification_types = Literal["", "react", "bats"]
dataset_types = Literal["Methoden", "hier", "einfügen"]
backbones = Literal["resnet50"]


def get_backbone(backbone: backbones):
    if backbone == "resnet50":
        return get_resnet50()


def run_experiment(backbone: backbones, layer:str, dataset:dataset_types, rectification_method:rectification_types, prune_method:prune_types):
    backbone = get_backbone(backbone)
    feature_extractor = FeatureExtractor(backbone, layer)
    #return appropriate methods
    rect_method, prune_method = get_methods(rectification=rectification_method, pruning=prune_method)

    pipeline = Pipeline(feature_extractor, rect_method, prune_method)
    
    dataloader = get_in_data_loader(dataset)[0]
    
    trainer = Trainer()

    trainer.test(pipeline, dataloaders=dataloader)

    return pipeline.energy_scores