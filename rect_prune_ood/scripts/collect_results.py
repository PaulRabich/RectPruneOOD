import sys

sys.path.append("./")
import os
from pandas import DataFrame
import pandas as pd
from tqdm import tqdm

from rect_prune_ood.lib.eval.metrics import compute_traditional_ood
from rect_prune_ood.lib.util.paths import get_top_level_path

datasets = [
    "imagenet_perturbed",
    "inat",
    "dtd",
    "sun",
    "places",
    "ninco",
    "ssb_hard",
]

# ood_datasets = [
#     "imagenet_perturbed",
#     "inat",
#     "dtd",
#     "sun",
#     "places",
#     "ninco",
#     "ssb_hard",
# ]
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
def build_table(
    results, 
    id_dataset="imagenet",
    rectification_method="",
    prune_method="",
    metric="energy_score",
):
    results.drop(columns=["id_dataset", "rectification_method", "prune_method", "score_metric"], inplace=True)
    results["FPR"] = results["FPR"] * 100
    results["AUROC"] = results["AUROC"] * 100
    results["DTERR"] = results["DTERR"] * 100
    results["AUIN"] = results["AUIN"] * 100
    results["AUOUT"] = results["AUOUT"] * 100

    string_table = ""
    rectification_method = "None" if rectification_method == "" else rectification_method
    prune_method = "None" if prune_method == "" else prune_method

    string_table += "ID_DATASET: " + id_dataset + "\n"
    string_table += "RECTIFICATION_METHOD: "+ rectification_method + "\n"
    string_table +="PRUNE_METHOD: " + prune_method + "\n"
    string_table +="METRIC: " + metric + "\n\n"
    
    # print the resutls in a nice table
    # Determine the maximum width for the ood_dataset column
    max_width = max(len("ood_datasets"), max(len(str(val)) for val in results["ood_dataset"])) + 2

    # Define consistent column widths for the numerical columns
    col_width = 10

    # Construct the header
    string_table += (
        f"{'ood_datasets':<{max_width}}"
        f"{'FPR':<{col_width}}"
        f"{'AUROC':<{col_width}}"
        f"{'DTERR':<{col_width}}"
        f"{'AUIN':<{col_width}}"
        f"{'AUOUT':<{col_width}}"
        "\n"
    )

    # Add a separator line
    string_table += "-" * (max_width + col_width * 5) + "\n"

    # Print rows with consistent widths
    for i in range(len(results)):
        string_table += (
            f"{results.iloc[i]['ood_dataset']:<{max_width}}"
            f"{results.iloc[i]['FPR']:<{col_width}.2f}"
            f"{results.iloc[i]['AUROC']:<{col_width}.2f}"
            f"{results.iloc[i]['DTERR']:<{col_width}.2f}"
            f"{results.iloc[i]['AUIN']:<{col_width}.2f}"
            f"{results.iloc[i]['AUOUT']:<{col_width}.2f}"
            "\n"
        )

    return string_table
def get_results_df():

    output = ""

    for id_dataset in tqdm(["imagenet", "imagenet_full_spec"]):

        ood_datasets = set(datasets.copy())
        if id_dataset == "imagenet_full_spec":
            ood_datasets.remove("imagenet_perturbed")

        ood_datasets = list(ood_datasets)
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
                    results = []

                    for i in range(len(ood_datasets)):
                        metric = value[i]
                        row =  {
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
                        results.append(row)


                    results = DataFrame(results)
                    table = build_table(results, id_dataset, rectification_method, prune_method, score)
                    output += table + "\n"
                

    with open(os.path.join(get_top_level_path(), "results", "scores.txt"), "w") as f:
        f.write(output)

if __name__ == "__main__":
    get_results_df()
