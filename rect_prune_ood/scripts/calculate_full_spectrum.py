
import sys
import os

sys.path.append("./")

import argparse
import numpy as np
import pandas as pd

from rect_prune_ood.lib.experiments.run_experiment import (
    run_full_spec_experiment,
    run_single_experiment,
)
from rect_prune_ood.lib.util.paths import get_top_level_path
from rect_prune_ood.lib.eval.metrics import cal_metric

#init paths
results_dir = os.path.join(get_top_level_path(), "results")
metrics_path = os.path.join(results_dir, "full_spectrum_scores.csv")

# Create results dir if needed
os.makedirs(results_dir, exist_ok=True)

# Initialize CSV if it doesn't exist
if not os.path.exists(metrics_path):
    pd.DataFrame(columns=["index", "method", "ood_dataset", "FPR", "AUROC", "DTERR", "AUIN", "AUOUT"]) \
        .set_index("index") \
        .to_csv(metrics_path)

for method_name in ["ashp_apes@95"]:# "apes@95", "react", "ashs", "ashp", "scale"
    run_full_spec_experiment(
        backbone="resnet50", 
        layer="avgpool",
        method_name=method_name,
    )

    for dataset in ["inat", "dtd", "sun", "places", "ninco", "ssb_hard"]:
        run_single_experiment(
            backbone="resnet50",
            layer="avgpool",
            method_name=method_name,
            capture_activations=False,
            dataset=dataset,
        )

        # Load energy scores
        full_spec_id = np.load(
            os.path.join(
                get_top_level_path(),
                f"results/imagenet/{method_name}/imagenet_full_spec.npy",
            ),
            allow_pickle=True,
        ).item()["energy_score"]

        ood_scores = np.load(
            os.path.join(
                get_top_level_path(),
                f"results/imagenet/{method_name}/{dataset}.npy",
            ),
            allow_pickle=True,
        ).item()["energy_score"]

        # Calculate metrics
        metrics = cal_metric(full_spec_id, ood_scores)

        # Load existing DataFrame to determine next index
        existing_df = pd.read_csv(metrics_path, index_col="index")
        next_index = len(existing_df)

        result_dict = {
            "index": next_index,
            "method": method_name,
            "ood_dataset": dataset,
            "FPR": metrics["FPR"],
            "AUROC": metrics["AUROC"],
            "DTERR": metrics["DTERR"],
            "AUIN": metrics["AUIN"],
            "AUOUT": metrics["AUOUT"],
        }

        print(result_dict)

        # Append result and save
        result_df = pd.DataFrame([result_dict]).set_index("index")
        result_df.to_csv(metrics_path, mode="a", header=False)
