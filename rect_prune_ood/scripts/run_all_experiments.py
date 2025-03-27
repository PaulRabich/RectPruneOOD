import sys
sys.path.append("./")

from rect_prune_ood.lib.experiments.run_experiment import run_all_experiments
from rect_prune_ood.scripts.collect_results import get_results_df

run_all_experiments(
    backbone="mobilenetV2",
    layer="avgpool",
    capture_activations=True,
    base_dataset="imagenet",
)

#get_results_df()
