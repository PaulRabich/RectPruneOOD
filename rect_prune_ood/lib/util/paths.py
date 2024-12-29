import os
import numpy as np

def get_top_level_path():
    # get the absolute path of this file
    abs_path = os.path.dirname(os.path.abspath(__file__))
    # get the parent directory
    top_level_path = os.path.abspath(os.path.join(abs_path, "../../../"))

    return top_level_path

def get_method_result_path(id_dataset: str = "imagenet", method_name: str = ""):
    if method_name == "":
        result_dir_name = "baseline"
    else:
        result_dir_name = method_name
        
    result_dir = os.path.join(
        get_top_level_path(),
        f"results/{id_dataset}/{result_dir_name}",
    )
    os.makedirs(result_dir, exist_ok=True)

    return result_dir