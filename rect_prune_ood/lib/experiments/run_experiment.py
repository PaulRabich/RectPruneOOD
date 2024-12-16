from typing import Literal
import os

import numpy as np
from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger

from rect_prune_ood.lib.experiments.feature_extractor import FeatureExtractor
from rect_prune_ood.lib.experiments.pipeline import Pipeline
from rect_prune_ood.lib.experiments.backbones import get_resnet50
from rect_prune_ood.lib.methods.rect_prune_methods import get_classes
from rect_prune_ood.lib.util.paths import get_method_result_path
from rect_prune_ood.lib.util.data_loader import (
    id_datasets, 
    ood_datasets, 
    perturbed_datasets, 
    get_data_loader
)


prune_types = Literal["", "ashb", "ashp", "ashs", "dice", "line"]
# weight_pruning_types= Literal["Methoden", "hier", "einfügen"]
rectification_types = Literal["", "react", "bats"]
backbones = Literal["resnet50"]


def get_backbone(backbone: backbones):
    if backbone == "resnet50":
        return get_resnet50()


def run_experiment(
    backbone: backbones,
    layer: str,
    id_dataset: id_datasets,
    ood_datasets: list[ood_datasets],
    perturbed_datasets: list[perturbed_datasets],
    rectification_method: rectification_types,
    prune_method: prune_types,
    capture_activations: bool,
):
    backbone = get_backbone(backbone)
    feature_extractor = FeatureExtractor(backbone, layer)

    # return appropriate methods
    rect_cls, prune_cls = get_classes(
        rectification=rectification_method, pruning=prune_method
    )

    rectification = rect_cls()
    pruning = prune_cls(fc_layer=feature_extractor.get_fc_layer())

    # get the pipeline and the dataloader
    pipeline = Pipeline(feature_extractor, rectification, pruning, caputure_activations=capture_activations)
    
    # generate the name/path for the result dir. The name should contain the methods and whether it is pertubated or not
    result_dir = get_method_result_path(id_dataset, rectification_method, prune_method)

    print(f'Evaluating id dataset "{id_dataset}"')
    id_dataloader = get_data_loader(id_dataset, batch_size=16)

    # Trainer
    trainer = Trainer(logger=None)
    trainer.test(pipeline, id_dataloader)

    id_result = pipeline.results

    np.save(os.path.join(result_dir, f"id_{id_dataset}.npy"), id_result)

    if capture_activations:
        np.save(
            os.path.join(result_dir, f"id_{id_dataset}_activations.npy"),
            pipeline.activations,
        )

    for perturbed_dataset in perturbed_datasets:
        print(f'Evaluating perturbed dataset "{perturbed_dataset}"')
        perturbation_dataloader = get_data_loader(perturbed_dataset, batch_size=16)

        trainer.test(pipeline, perturbation_dataloader)

        pertubation_result = pipeline.results

        np.save(os.path.join(result_dir, f"{perturbed_dataset}.npy"), pertubation_result)

    for ood_dataset in ood_datasets:
        print(f'Evaluating ood dataset "{ood_dataset}"')

        ood_dataloader = get_data_loader(ood_dataset, batch_size=16)

        trainer.test(pipeline, ood_dataloader)

        ood_result = pipeline.results

        np.save(os.path.join(result_dir, f"ood_{ood_dataset}.npy"), ood_result)
        if capture_activations:
            np.save(
                os.path.join(result_dir, f"ood_{ood_dataset}_activations.npy"),
                pipeline.activations,
            )
