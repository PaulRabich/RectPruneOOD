import sys
sys.path.append("./")

from rect_prune_ood.lib.experiments.run_experiment import run_all_experiments

run_all_experiments(
    backbone="resnet50",
    layer="avgpool",
    capture_activations=False,
    id_dataset="imagenet",
)