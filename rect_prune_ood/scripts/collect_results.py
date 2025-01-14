import sys

sys.path.append("./")
import os
from pandas import DataFrame
import pandas as pd
from tqdm import tqdm

from rect_prune_ood.lib.eval.metrics import compute_traditional_ood
from rect_prune_ood.lib.util.paths import get_top_level_path

datasets = [
    "imagenet",
    "imagenet_perturbed",
    "inat",
    "dtd",
    "sun",
    "places",
    "ninco",
    "ssb_hard",
]

ood_datasets = [
    "imagenet_perturbed",
    "inat",
    "dtd",
    "sun",
    "places",
    "ninco",
    "ssb_hard",
]
rectification_methods = ["react", ""]
prune_methods = ["ashb", "ashp", "ashs", ""]
scores = ["energy_score", "msp_score", "entropy_score", "confidence"]

columns = [
    "id_dataset",
    "ood_dataset",
    "rectification_method",
    "prune_method",
    "score_metric",
    "FPR",
    "AUROC",
    "DTERR",
    "AUIN", 
    "AUOUT",
]

def get_results_df():

    result_df = DataFrame(columns=columns)

    for id_dataset in tqdm(["imagenet", "imagenet_full_spec"]):
        for rectification_method in rectification_methods:
            for prune_method in prune_methods:
                for score in scores:
                    value = compute_traditional_ood(
                        id_dataset=id_dataset,
                        ood_datasets=ood_datasets,
                        rectification_method=rectification_method,
                        prune_method=prune_method,
                        score_metric=score,
                        print_results=False,
                    )
                    for i in range(len(ood_datasets)):
                        metric = value[i]
                        row = DataFrame(
                            [
                                {
                                    "id_dataset": id_dataset,
                                    "ood_dataset": ood_datasets[i],
                                    "rectification_method": rectification_method,
                                    "prune_method": prune_method,
                                    "score_metric": score,
                                    "FPR": metric["FPR"],
                                    "AUROC": metric["AUROC"],
                                    "DTERR": metric["DTERR"],
                                    "AUIN": metric["AUIN"],
                                    "AUOUT": metric["AUOUT"],
                                }
                            ]
                        )
                        result_df = pd.concat([result_df, row])

    score_path =os.path.join(get_top_level_path(),"results","scores")
    os.makedirs(score_path, exist_ok=True)
    result_df.to_csv(os.path.join(score_path, "collected_results.csv"), index=False)

if __name__ == "__main__":
    get_results_df()
