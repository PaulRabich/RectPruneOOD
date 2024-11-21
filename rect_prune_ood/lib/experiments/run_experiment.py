from typing import Literal
import os

import numpy as np
from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger

from rect_prune_ood.lib.experiments.feature_extractor import FeatureExtractor
from rect_prune_ood.lib.experiments.pipeline import Pipeline
from rect_prune_ood.lib.experiments.backbones import get_resnet50
from rect_prune_ood.lib.methods.rect_prune_methods import get_classes
from rect_prune_ood.lib.util import get_in_data_loader, get_out_data_loader, get_top_level_path


prune_types = Literal["", "ashb", "ashp", "ashs", "dice", "line"]
#weight_pruning_types= Literal["Methoden", "hier", "einfügen"]
rectification_types = Literal["", "react", "bats"]
dataset_types = Literal["cifar10"]
backbones = Literal["resnet50"]


def get_backbone(backbone: backbones):
    if backbone == "resnet50":
        return get_resnet50()

 


def run_experiment(backbone: backbones, layer:str, dataset:dataset_types, rectification_method:rectification_types, prune_method:prune_types):
    backbone = get_backbone(backbone)
    feature_extractor = FeatureExtractor(backbone, layer)

    #return appropriate methods
    rect_cls, prune_cls= get_classes(rectification=rectification_method, pruning=prune_method)

    rect_method = rect_cls()
    prune_method = prune_cls(fc_layer=feature_extractor.get_fc_layer())
    
    # get the pipeline and the dataloader
    pipeline = Pipeline(feature_extractor, rect_method, prune_method)
    dataloader, num_classes = get_in_data_loader(dataset, shuffle=False, batch_size=16)

    # Trainer
    trainer = Trainer(logger=None)
    trainer.test(pipeline, dataloader)

    res = pipeline.results

    np.save(os.path.join(get_top_level_path(), "results", "results.npy"), res)
    return res

    