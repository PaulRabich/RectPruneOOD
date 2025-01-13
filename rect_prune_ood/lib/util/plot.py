import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from rect_prune_ood.lib.util.paths import get_method_result_path
from rect_prune_ood.lib.eval.metrics import cal_metric, compute_average_results

def plot_compare_methods(
        ood_datasets: Union[str, list[str]], 
        id_dataset: str = "imagenet",
        first_method: str = "baseline",
        second_method: str = "react",
        score_function: str = "energy_score",
        plot_iou: bool = False,
        plot_auroc: bool = False,
        use_full_spec: bool = True,
        scale_y: bool = False
    ):
    """
    Plots the comparison of two methods for a specified ID dataset and OOD datasets.

    Example call for plotting the comparison of the baseline and ReAct methods for the ImageNet dataset and the iNaturalist, DTD, SUN, and Places365 OOD datasets:

    plot_compare_methods(["inat", "dtd", "sun", "places"], "imagenet", "baseline", "react", "energy_score", plot_iou=True, plot_auroc=True, use_full_spec=True, scale_y=False)

    Args:
    ood_datasets (Union[str, list[str]]): Name of the OOD datasets. Can either be a single entry (string) or multiple (list of strings).
    id_dataset (str): Name of the ID dataset.
    first_method (str): Name of the first method.
    second_method (str): Name of the second method.
    score_function (str): Name of the score function.
    plot_iou (bool): Whether to plot the IoU.
    plot_auroc (bool): Whether to plot the AUROC.
    use_full_spec (bool): Whether to use the full spectrum of the ID dataset (ImageNet + ImageNet perturbed).
    scale_y (bool): Whether to match the scale of the y-axis for both methods.
    """
    fm_scores = load_scores_for_method(first_method, id_dataset, score_function)
    sm_scores = load_scores_for_method(second_method, id_dataset, score_function)

    if use_full_spec:
        fm_id_scores = fm_scores[f"{id_dataset}_full_spec"]
        sm_id_scores = sm_scores[f"{id_dataset}_full_spec"]
    else:
        fm_id_scores = fm_scores[f"{id_dataset}"]
        sm_id_scores = sm_scores[f"{id_dataset}"]

    fm_ood_scores = []
    sm_ood_scores = []

    if plot_auroc:
        fm_metrics = []
        sm_metrics = []

    ood_datasets_str = ""

    for ood_dataset in ood_datasets:
        fm_ood_scores.extend(fm_scores[ood_dataset])
        sm_ood_scores.extend(sm_scores[ood_dataset])

        ood_datasets_str += f"{ood_dataset}, "

        if plot_auroc:
            fm_metrics.append(cal_metric(fm_id_scores, fm_scores[ood_dataset]))
            sm_metrics.append(cal_metric(sm_id_scores, sm_scores[ood_dataset]))

    ood_datasets_str = ood_datasets_str[:-2]

    if plot_auroc:
        fm_metrics = compute_average_results(fm_metrics)
        sm_metrics = compute_average_results(sm_metrics)

    fm_bin_range = (
        min(min(fm_id_scores), min(fm_ood_scores)),
        max(max(fm_id_scores), max(fm_ood_scores))
    )

    sm_bin_range = (
        min(min(sm_id_scores), min(sm_ood_scores)),
        max(max(sm_id_scores), max(sm_ood_scores))
    )

    # Histograms for the first plot
    fm_id_hist, fm_bin_edges = np.histogram(fm_id_scores, bins=1000, range=fm_bin_range, density=True)
    fm_ood_hist, _ = np.histogram(fm_ood_scores, bins=1000, range=fm_bin_range, density=True)

    # Histograms for the second plot
    sm_id_hist, sm_bin_edges = np.histogram(sm_id_scores, bins=1000, range=sm_bin_range, density=True)
    sm_ood_hist, _ = np.histogram(sm_ood_scores, bins=1000, range=sm_bin_range, density=True)

    # Midpoints of bins for plotting
    fm_bin_centers = (fm_bin_edges[:-1] + fm_bin_edges[1:]) / 2
    sm_bin_centers = (sm_bin_edges[:-1] + sm_bin_edges[1:]) / 2

    # Two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Beautify names
    fm_name = _beautify_method_name(first_method)
    sm_name = _beautify_method_name(second_method)

    id_dataset_name = _beautify_dataset_name(id_dataset)
    score_function_name = _beautify_score_function_name(score_function)

    # Plot 1
    axs[0].fill_between(fm_bin_centers, 0, fm_id_hist, color='blue', alpha=0.3, label=f"{fm_name} ID")
    axs[0].fill_between(fm_bin_centers, 0, fm_ood_hist, color='red', alpha=0.3, label=f"{fm_name} OOD")
    axs[0].plot(fm_bin_centers, fm_id_hist, color='blue', alpha=0.6, linewidth=0.2)
    axs[0].plot(fm_bin_centers, fm_ood_hist, color='red', alpha=0.6, linewidth=0.2)

    # Plot IoU
    if plot_iou:
        iou, intersection, _ = compute_iou(fm_id_hist, fm_ood_hist)

        axs[0].fill_between(fm_bin_centers, 0, intersection, color='darkgray', alpha=0.5, label=f"IoU (↓) = {iou:.4f}")

    # Plot AUROC
    if plot_auroc:
        axs[0].plot(min(fm_bin_range), 0, color='none', label=f'AUROC (↑) = {100.*fm_metrics["AUROC"]:6.2f}')

    axs[0].set_title(f"{fm_name}: ID vs OOD ({ood_datasets_str})")

    axs[0].set_xlabel(score_function_name)
    axs[0].set_ylabel("Frequency")
    axs[0].legend()

    # Plot 2
    axs[1].fill_between(sm_bin_centers, 0, sm_id_hist, color='blue', alpha=0.3, label=f"{sm_name} ID")
    axs[1].fill_between(sm_bin_centers, 0, sm_ood_hist, color='red', alpha=0.3, label=f"{sm_name} OOD")
    axs[1].plot(sm_bin_centers, sm_id_hist, color='blue', alpha=0.6, linewidth=0.2)
    axs[1].plot(sm_bin_centers, sm_ood_hist, color='red', alpha=0.6, linewidth=0.2)

    # Plot IoU
    if plot_iou:
        iou, intersection, _ = compute_iou(sm_id_hist, sm_ood_hist)

        axs[1].fill_between(sm_bin_centers, 0, intersection, color='darkgray', alpha=0.5, label=f"IoU (↓) = {iou:.4f}")

    # Plot AUROC
    if plot_auroc:
        axs[1].plot(min(sm_bin_range), 0, color='none', label=f'AUROC (↑) = {100.*sm_metrics["AUROC"]:6.2f}')

    axs[1].set_title(f"{sm_name}: ID vs OOD ({ood_datasets_str})")
    axs[1].set_xlabel(score_function_name)
    axs[1].set_ylabel("Frequency")
    axs[1].legend()

    # scale y axis to the same as the first plot
    if scale_y:
        axs[1].set_ylim(axs[0].get_ylim())

    fig.suptitle(f"{fm_name} vs {sm_name}: {score_function_name} comparison for ID dataset {id_dataset_name}", fontsize=16)
    fig.subplots_adjust(top=0.85)

def load_scores_for_method(method_name: str = "baseline", id_dataset: str = "imagenet", score_function: str = "energy_score") -> dict:
    """
    Loads all scores for a specified method and ID dataset.

    Args:
    method_name (str): Name of the method.
    id_dataset (str): Name of the ID dataset. Only used for folder structure.
    score_function (str): Name of the score function.

    Returns:
    dict: Dictionary containing the scores for the specified method for all datasets.
    """
    method_result_path = get_method_result_path(id_dataset, method_name)
    
    scores = {}

    # Loop over all files in the method result directory
    for file_name in os.listdir(method_result_path):
        if "activations" in file_name:
            continue
        scores[file_name[:-4]] = np.load(os.path.join(method_result_path, file_name), allow_pickle=True).item()[score_function]

    return scores

def _beautify_method_name(method_name: str = "") -> str:
    if method_name == "baseline":
        return "Baseline"
    elif method_name == "react":
        return "ReAct"
    elif method_name == "bats":
        return "BATS"
    elif method_name == "ashs":
        return "AshS"
    elif method_name == "ashb":
        return "AshB"
    elif method_name == "ashp":
        return "AshP"
    elif method_name == "dice":
        return "Dice"
    elif method_name == "line":
        return "Line"
    elif method_name == "react_ashs":
        return "ReAct + AshS"
    elif method_name == "react_ashb":
        return "ReAct + AshB"
    elif method_name == "react_ashp":
        return "ReAct + AshP"
    elif method_name == "react_dice":
        return "ReAct + Dice"
    elif method_name == "react_line":
        return "ReAct + Line"

    return method_name

def _beautify_dataset_name(dataset_name: str = "") -> str:
    if dataset_name == "imagenet":
        return "ImageNet"
    elif dataset_name == "inat":
        return "iNaturalist"
    elif dataset_name == "dtd":
        return "DTD"
    elif dataset_name == "sun":
        return "SUN"
    elif dataset_name == "places":
        return "Places365"
    elif dataset_name == "ninco":
        return "NINCO"
    elif dataset_name == "ssb_hard":
        return "SSB Hard"
    elif dataset_name == "imagenet_perturbed":
        return "ImageNet Perturbed"
    elif dataset_name == "imagenet_full_spec":
        return "ImageNet Full Spec"

    return dataset_name

def _beautify_score_function_name(score_function: str = "") -> str:
    if score_function == "energy_score":
        return "Energy Score"
    elif score_function == "msp_score":
        return "MSP Score"
    elif score_function == "entropy_score":
        return "Entropy Score"
    else:
        return "Confidence"

def compute_iou_for_array(data_1: np.array, data_2: np.array, num_bins: int = 1000) -> tuple[float, np.array, np.array]:
    """
    Compute the Intersection over Union (IoU) of two arrays.

    Args:
    data_1 (np.array): First array.
    data_2 (np.array): Second array.
    num_bins (int): Number of bins for the histograms.

    Returns:
    float: Intersection over Union (IoU) of the arrays.
    np.array: Intersection of the arrays.
    np.array: Union of the arrays.
    """

    # Determine range of the histograms
    max_val = max(max(data_1), max(data_2))
    min_val = min(min(data_1), min(data_2))
    bin_range = (min_val, max_val)

    # Calculate histograms and bin edges
    data_1_hist, _ = np.histogram(data_1, bins=num_bins, range=bin_range, density=True)
    data_2_hist, _ = np.histogram(data_2, bins=num_bins, range=bin_range, density=True)

    return compute_iou(data_1_hist, data_2_hist)


def compute_iou(hist_1: np.histogram, hist_2: np.histogram) -> tuple[float, np.array, np.array]:
    """
    Compute the Intersection over Union (IoU) of two histograms.

    Args:
    hist_1 (np.histogram): First histogram.
    hist_2 (np.histogram): Second histogram.

    Returns:
    float: Intersection over Union (IoU) of the histograms.
    np.array: Intersection of the histograms.
    np.array: Union of the histograms.
    """

    # Intersection over Union (IoU)
    intersection = np.minimum(hist_1, hist_2)
    union = np.maximum(hist_1, hist_2)
    iou = np.sum(intersection) / np.sum(union)

    return iou, intersection, union