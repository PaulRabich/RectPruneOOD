from typing import Literal
import os

import numpy as np
from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger

from rect_prune_ood.lib.experiments.feature_extractor import FeatureExtractor
from rect_prune_ood.lib.experiments.pipeline import Pipeline
from rect_prune_ood.lib.experiments.backbones import get_resnet50
from rect_prune_ood.lib.methods.rect_prune_methods import get_classes
from rect_prune_ood.lib.util import get_id_data_loader, get_ood_data_loader, get_top_level_path
from rect_prune_ood.lib.util.data_loader import id_datasets, ood_datasets


prune_types = Literal["", "ashb", "ashp", "ashs", "dice", "line"]
#weight_pruning_types= Literal["Methoden", "hier", "einfügen"]
rectification_types = Literal["", "react", "bats"]
backbones = Literal["resnet50"]


def get_backbone(backbone: backbones):
    if backbone == "resnet50":
        return get_resnet50()

 


def run_experiment(backbone: backbones, layer:str, id_dataset: id_datasets, ood_datasets:list[ood_datasets], rectification_method:rectification_types, prune_method:prune_types):
    backbone = get_backbone(backbone)
    feature_extractor = FeatureExtractor(backbone, layer)

    #return appropriate methods
    rect_cls, prune_cls= get_classes(rectification=rectification_method, pruning=prune_method)

    rect_method = rect_cls()
    prune_method = prune_cls(fc_layer=feature_extractor.get_fc_layer())
    
    # get the pipeline and the dataloader
    pipeline = Pipeline(feature_extractor, rect_method, prune_method)

    result_dir = os.path.join(get_top_level_path(), f"results/{id_dataset}") 
    os.makedirs(result_dir, exist_ok=True)

    print(f"Evaluating id dataset \"{id_dataset}\"")
    dataloader, _ = get_id_data_loader(id_dataset, batch_size=16)

    # Trainer
    trainer = Trainer(logger=None)
    trainer.test(pipeline, dataloader)

    id_result = pipeline.results        

    np.save(os.path.join(result_dir, f"id_{id_dataset}.npy"), id_result)
    np.save(os.path.join(result_dir, f"id_{id_dataset}_activations.npy"), pipeline.activations)

    for ood_dataset in ood_datasets:
        print(f"Evaluating ood dataset \"{ood_dataset}\"")

        dataloader = get_ood_data_loader(
            ood_dataset,
            batch_size=16
        )

        trainer.test(pipeline, dataloader)

        ood_result = pipeline.results

        np.save(os.path.join(result_dir, f"ood_{ood_dataset}.npy"), ood_result)
        np.save(os.path.join(result_dir, f"ood_{ood_dataset}_activations.npy"), pipeline.activations)

    