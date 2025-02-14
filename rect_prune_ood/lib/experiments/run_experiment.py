from typing import Literal, Union
import os

import numpy as np
from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger

from rect_prune_ood.lib.experiments.feature_extractor import FeatureExtractor
from rect_prune_ood.lib.experiments.mobilenet_feature_extractor import MobileNetFeatureExtractor
from rect_prune_ood.lib.experiments.pipeline import Pipeline
from rect_prune_ood.lib.experiments.backbones import get_resnet50, get_mobilenetV2
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
backbones = Literal["resnet50", "mobilenetV2"]


def get_feature_extractor(backbone: backbones, layer: str):
    if backbone == "mobilenetV2":
        return MobileNetFeatureExtractor(get_mobilenetV2())
    else:
        return FeatureExtractor(get_resnet50(), layer)

def get_backbone(backbone: backbones):
    if backbone == "resnet50":
        return get_resnet50()
    
def run_all_experiments(
    backbone: backbones,
    layer: str,
    capture_activations: bool = False,
    base_dataset: id_datasets = "imagenet",
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
                base_dataset=base_dataset,
                mask_args=mask_args,
            )
            run_full_spec_experiment(
                backbone=backbone,
                layer=layer,
                rectification_method=rectification_method,
                prune_method=prune_method,
                capture_activations=capture_activations,
                mask_args=mask_args,
            )

def run_experiment_for_method(
    backbone: backbones,
    layer: str,
    method_name:str,
    capture_activations: bool = False,
    base_dataset: id_datasets = "imagenet",
):
    for dataset in [
        'imagenet',
        #'imagenet_perturbed',
        'inat',
        'dtd',
        'sun',
        'places',
        'ninco',
        'ssb_hard',
        'openimage-o'
    ]:
        run_single_experiment(
            backbone=backbone,
            layer=layer,
            dataset=dataset,
            method_name=method_name,
            capture_activations=capture_activations,
            base_dataset=base_dataset,
        )
    
def run_single_experiment(
    backbone: backbones,
    layer: str,
    dataset: Union[id_datasets, ood_datasets, perturbed_datasets],
    method_name:str,
    capture_activations: bool = False,
    base_dataset: id_datasets = "imagenet",
):
    """
    Runs a single experiment for a specified dataset and method. Saves the results in the corresponding directory.

    Args:
    backbone (backbones): Backbone to use.
    layer (str): Layer to use.
    dataset (Union[id_datasets, ood_datasets, perturbed_datasets]): Dataset for which to run the experiment.
    method_name (str): Name of the method.
    capture_activations (bool): Whether to capture the activations.
    base_dataset (id_datasets): Base dataset. Mainly used for paths. Defaults to "imagenet".
    """
    #backbone = get_backbone(backbone)
    feature_extractor = get_feature_extractor(backbone, layer)
    
    fc_layer = fc_layer=feature_extractor.get_fc_layer()

    # get the pipeline and the dataloader
    pipeline = Pipeline(feature_extractor, method_name, fc_layer, caputure_activations=capture_activations)

    result_dir = get_method_result_path(base_dataset, method_name)

    # print(f'Evaluating dataset "{dataset}" for rectification "{rectification_method}" and pruning "{prune_method}"')
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
    method_name:str,
    capture_activations: bool = False,
):
    """
    Runs the experiment for both datasets, imagenet and imagenet_perturbed and concatenates the results. The results are saved in the corresponding directory.

    Args:
    backbone (backbones): Backbone to use.
    layer (str): Layer to use.
    method_name (str): Name of the method.
    capture_activations (bool): Whether to capture the activations.
    mask_args (dict): Arguments for the mask.

    Returns:
    None
    """
    base_dataset = "imagenet"
    backbone = get_backbone(backbone)
    feature_extractor = FeatureExtractor(backbone, layer)

    fc_layer=feature_extractor.get_fc_layer()

    # get the pipeline and the dataloader
    pipeline = Pipeline(feature_extractor, method_name, fc_layer, caputure_activations=capture_activations)
        
    result_dir = get_method_result_path(base_dataset, method_name)

    print(f'Creating full spectrum results for "{base_dataset}" with method "{method_name}"')
    
    dataloader, _ = get_data_loader(base_dataset, batch_size=16)

    # Trainer
    trainer = Trainer(logger=None)
    trainer.test(pipeline, dataloader)

    results_normal = pipeline.results
    if capture_activations:
        activations_normal = pipeline.activations

    pert_dataloader, _ = get_data_loader(f"{base_dataset}_perturbed", batch_size=16)
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

    np.save(os.path.join(result_dir, f"{base_dataset}_full_spec.npy"), results)

    if capture_activations:
        np.save(
            os.path.join(result_dir, f"{base_dataset}_full_spec_activations.npy"),
            activations,
        )

