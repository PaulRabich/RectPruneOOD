import os
import numpy as np

def get_top_level_path():
    # get the absolute path of this file
    abs_path = os.path.dirname(os.path.abspath(__file__))
    # get the parent directory
    top_level_path = os.path.abspath(os.path.join(abs_path, "../../../"))

    return top_level_path

def get_method_result_path(id_dataset: str = "imagenet", pertubated: bool = False, rectification_method: str = "", prune_method: str = ""):
    if rectification_method == "" and prune_method == "":
        result_dir_name = "baseline"
    elif rectification_method == "":
        result_dir_name = f"{prune_method}"
    elif prune_method == "":
        result_dir_name = f"{rectification_method}"
    else:
        result_dir_name = f"{rectification_method}_{prune_method}".strip("_")

    if pertubated:
        result_dir_name += "_pertubated"

    result_dir = os.path.join(
        get_top_level_path(),
        f"results/{id_dataset}/{result_dir_name}",
    )
    os.makedirs(result_dir, exist_ok=True)

    return result_dir