import sys
sys.path.append("./")

from rect_prune_ood.lib.experiments.run_experiment import run_all_experiments
from rect_prune_ood.scripts.collect_results import get_results_df

run_all_experiments(
    backbone="resnet50",
    layer="avgpool",
    capture_activations=False,
    id_dataset="imagenet",
)

get_results_df()
