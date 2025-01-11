from typing import Literal, Union
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
    
def run_all_experiments(
    backbone: backbones,
    layer: str,
    capture_activations: bool = False,
    id_dataset: id_datasets = "imagenet",
    mask_args=None,
):
    for rectification_method in ["react", ""]:
        for prune_method in ["ashb", "ashp", "ashs", ""]:
            run_experiment_for_method(
                backbone=backbone,
                layer=layer,
                rectification_method=rectification_method,
                prune_method=prune_method,
                capture_activations=capture_activations,
                id_dataset=id_dataset,
                mask_args=mask_args,
            )
            run_full_spec_experiment(
                backbone=backbone,
                layer=layer,
                rectification_method=rectification_method,
                prune_method=prune_method,
                capture_activations=capture_activations,
                id_dataset=id_dataset,
                mask_args=mask_args,
            )

def run_experiment_for_method(
    backbone: backbones,
    layer: str,
    rectification_method: rectification_types,
    prune_method: prune_types,
    capture_activations: bool = False,
    id_dataset: id_datasets = "imagenet",
    mask_args=None,
):
    for dataset in [
        'imagenet',
        'imagenet_perturbed',
        'inat',
        'dtd',
        'sun',
        'places',
        'ninco',
        'ssb_hard'
    ]:
        run_single_experiment(
            backbone=backbone,
            layer=layer,
            dataset=dataset,
            rectification_method=rectification_method,
            prune_method=prune_method,
            capture_activations=capture_activations,
            id_dataset=id_dataset,
            mask_args=mask_args,
        )
    
def run_single_experiment(
    backbone: backbones,
    layer: str,
    dataset: Union[id_datasets, ood_datasets, perturbed_datasets],
    rectification_method: rectification_types,
    prune_method: prune_types,
    capture_activations: bool = False,
    id_dataset: id_datasets = "imagenet",
    mask_args=None,
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
    pipeline = Pipeline(feature_extractor, rectification, pruning, caputure_activations=capture_activations, mask_args=mask_args)

    # generate the name/path for the result dir. The name should contain the methods and whether it is pertubated or not
    if rectification_method and prune_method:
        folder_name = f"{rectification_method}_{prune_method}"
    elif rectification_method:
        folder_name = rectification_method
    elif prune_method:
        folder_name = prune_method
    else:
        folder_name = "baseline"

    result_dir = get_method_result_path(id_dataset, folder_name)

    print(f'Evaluating dataset "{dataset}" for rectification "{rectification_method}" and pruning "{prune_method}"')
    dataloader, _ = get_data_loader(dataset, batch_size=16)

    # Trainer
    trainer = Trainer(logger=None)
    trainer.test(pipeline, dataloader)

    np.save(os.path.join(result_dir, f"{dataset}.npy"), pipeline.results)

    if capture_activations:
        np.save(
            os.path.join(result_dir, f"{dataset}_activations.npy"),
            pipeline.activations,
        )

def run_full_spec_experiment(
    backbone: backbones,
    layer: str,
    rectification_method: rectification_types,
    prune_method: prune_types,
    capture_activations: bool = False,
    id_dataset: id_datasets = "imagenet",
    mask_args=None,
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
    pipeline = Pipeline(feature_extractor, rectification, pruning, caputure_activations=capture_activations, mask_args=mask_args)

    # generate the name/path for the result dir. The name should contain the methods and whether it is pertubated or not
    if rectification_method and prune_method:
        folder_name = f"{rectification_method}_{prune_method}"
    elif rectification_method:
        folder_name = rectification_method
    elif prune_method:
        folder_name = prune_method
    else:
        folder_name = "baseline"
        
    result_dir = get_method_result_path(id_dataset, folder_name)

    print(f'Creating full spectrum results for "{id_dataset}"')
    dataloader, _ = get_data_loader(id_dataset, batch_size=16)

    # Trainer
    trainer = Trainer(logger=None)
    trainer.test(pipeline, dataloader)

    results_normal = pipeline.results
    if capture_activations:
        activations_normal = pipeline.activations

    pert_dataloader, _ = get_data_loader(f"{id_dataset}_perturbed", batch_size=16)
    trainer.test(pipeline, pert_dataloader)

    results_pert = pipeline.results
    if capture_activations:
        activations_pert = pipeline.activations

    results = {
        "energy_score": np.concatenate([results_normal["energy_score"], results_pert["energy_score"]], axis=0),
        "msp_score": np.concatenate([results_normal["msp_score"], results_pert["msp_score"]], axis=0),
        "entropy_score": np.concatenate([results_normal["entropy_score"], results_pert["entropy_score"]], axis=0),
        "prediction": np.concatenate([results_normal["prediction"], results_pert["prediction"]], axis=0),
        "label": np.concatenate([results_normal["label"], results_pert["label"]], axis=0),
        "confidence": np.concatenate([results_normal["confidence"], results_pert["confidence"]], axis=0),
    }

    if capture_activations:
        activations = np.concatenate([activations_normal, activations_pert], axis=0)

    np.save(os.path.join(result_dir, f"{id_dataset}_full_spec.npy"), results)

    if capture_activations:
        np.save(
            os.path.join(result_dir, f"{id_dataset}_full_spec_activations.npy"),
            activations,
        )

"""
This function is DEPRECATED. Use run_single_experiment instead.
"""
def run_experiment(
    backbone: backbones,
    layer: str,
    id_dataset: id_datasets,
    ood_datasets: list[ood_datasets],
    perturbed_datasets: list[perturbed_datasets],
    rectification_method: rectification_types,
    prune_method: prune_types,
    capture_activations: bool,
    mask_args=None,
):
    """
    DEPRECATED
    """

    backbone = get_backbone(backbone)
    feature_extractor = FeatureExtractor(backbone, layer)

    # return appropriate methods
    rect_cls, prune_cls = get_classes(
        rectification=rectification_method, pruning=prune_method
    )

    rectification = rect_cls()
    pruning = prune_cls(fc_layer=feature_extractor.get_fc_layer())

    # get the pipeline and the dataloader
    pipeline = Pipeline(feature_extractor, rectification, pruning, caputure_activations=capture_activations, mask_args=mask_args)
    
    # generate the name/path for the result dir. The name should contain the methods and whether it is pertubated or not
    if rectification_method and prune_method:
        folder_name = f"{rectification_method}_{prune_method}"
    elif rectification_method:
        folder_name = rectification_method
    elif prune_method:
        folder_name = prune_method
    result_dir = get_method_result_path(id_dataset, folder_name)

    print(f'Evaluating id dataset "{id_dataset}"')
    id_dataloader, _ = get_data_loader(id_dataset, batch_size=16)

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
        perturbation_dataloader, _ = get_data_loader(perturbed_dataset, batch_size=16)

        trainer.test(pipeline, perturbation_dataloader)

        pertubation_result = pipeline.results

        np.save(os.path.join(result_dir, f"id_{perturbed_dataset}.npy"), pertubation_result)

    for ood_dataset in ood_datasets:
        print(f'Evaluating ood dataset "{ood_dataset}"')

        ood_dataloader, _ = get_data_loader(ood_dataset, batch_size=16)

        trainer.test(pipeline, ood_dataloader)

        ood_result = pipeline.results

        np.save(os.path.join(result_dir, f"ood_{ood_dataset}.npy"), ood_result)
        if capture_activations:
            np.save(
                os.path.join(result_dir, f"ood_{ood_dataset}_activations.npy"),
                pipeline.activations,
            )
