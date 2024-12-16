import os
import numpy as np
import matplotlib.pyplot as plt
from rect_prune_ood.lib.util.paths import get_method_result_path , get_top_level_path

def plot_score_distributions(data_1: np.array, data_2: np.array, title: str, x_label: str, y_label: str, data_1_label: str, data_2_label: str, bin_range=None, num_bins: int = 1000, plot_iou: bool = False) -> None:
    """
    Plot the score distributions of two datasets and calculate the Intersection over Union (IoU) of the histograms.

    Args:
    data_1 (np.array): Scores of the first dataset.
    data_2 (np.array): Scores of the second dataset.
    title (str): Title of the plot.
    x_label (str): Label for the x-axis.
    y_label (str): Label for the y-axis.
    data_1_label (str): Label for the first dataset.
    data_2_label (str): Label for the second dataset.
    bin_range (tuple): Range of the histograms. If None, the range is determined by the minimum and maximum values of the datasets.
    num_bins (int): Number of bins for the histograms.
    plot_iou (bool): If True, the Intersection over Union (IoU) of the histograms is calculated and displayed.

    Returns:
    None
    """
    data_1_color = 'blue'
    data_2_color = 'red'

    # Determine range of the histograms
    max_val = max(max(data_1), max(data_2))
    min_val = min(min(data_1), min(data_2))
    bin_range = (min_val, max_val) if bin_range is None else bin_range

    # Calculate histograms and bin edges
    data_1_hist, bin_edges = np.histogram(data_1, bins=num_bins, range=bin_range, density=True)
    data_2_hist, _ = np.histogram(data_2, bins=num_bins, range=bin_range, density=True)

    plt.figure(figsize=(10, 6))

    # Midpoints of bins for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    plt.fill_between(bin_centers, 0, data_1_hist, color=data_1_color, alpha=0.3, label=data_1_label)
    
    plt.fill_between(bin_centers, 0, data_2_hist, color=data_2_color, alpha=0.3, label=data_2_label)

    plt.plot(bin_centers, data_1_hist, color=data_1_color, alpha=0.6, linewidth=0.2)
    plt.plot(bin_centers, data_2_hist, color=data_2_color, alpha=0.6, linewidth=0.2)

    if plot_iou:
        overlap_color = 'darkgray'

        # Intersection over Union (IoU)
        intersection = np.minimum(data_1_hist, data_2_hist)
        union = np.maximum(data_1_hist, data_2_hist)
        iou = np.sum(intersection) / np.sum(union)

        # Fill overlap area
        plt.fill_between(bin_centers, 0, intersection, color=overlap_color, alpha=0.5, label="Overlap")

        # Add IoU text to the plot
        plt.text(0.7 * bin_range[1] + 0.3 * bin_range[0], max(max(data_1_hist), max(data_2_hist)) * 0.9, 
                f"IoU = {iou:.4f}", fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.5))

    # Add labels, title, and legend
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()

def plot_perturbed_comparison(id_dataset: str = "imagenet", method_name: str = "baseline", score_function: str = "energy_score", num_bins: int = 1000, show_title: bool = True) -> None:
    # Load the results
    result_dir = get_method_result_path(id_dataset, method_name)

    id_scores = np.load(os.path.join(result_dir, f"id_{id_dataset}.npy"), allow_pickle=True).item()[score_function]

    id_scores_perturbed = np.load(os.path.join(result_dir, f"id_{id_dataset}_perturbed.npy"), allow_pickle=True).item()[score_function]

    ood_scores = []

    for ood_dataset_ in ['inat', 'dtd', 'sun', 'places']:
        ood_dataset_scores = np.load(os.path.join(result_dir, f"ood_{ood_dataset_}.npy"), allow_pickle=True).item()[score_function]
        ood_scores.append(ood_dataset_scores)

    ood_scores = np.concatenate(ood_scores)
        

    # Beautify names
    id_dataset_name, _, score_function_name, method_name_b = _beautify_names(id_dataset, "", score_function, method_name)

    # Make 
    plot_1_bin_range = (
        min(min(id_scores), min(ood_scores)),
        max(max(id_scores), max(ood_scores))
    )
    plot_2_bin_range = (
        min(min(id_scores), min(id_scores_perturbed)),
        max(max(id_scores), max(id_scores_perturbed))
    )

    # Histograms for the first plot
    id_1_hist, bin_edges = np.histogram(id_scores, bins=num_bins, range=plot_1_bin_range, density=True)
    ood_hist, _ = np.histogram(ood_scores, bins=num_bins, range=plot_1_bin_range, density=True)

    # Histograms for the second plot
    id_2_hist, _ = np.histogram(id_scores, bins=num_bins, range=plot_2_bin_range, density=True)
    id_pertubated_hist, _ = np.histogram(id_scores_perturbed, bins=num_bins, range=plot_2_bin_range, density=True)

    # Midpoints of bins for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1
    axs[0].fill_between(bin_centers, 0, id_1_hist, color='blue', alpha=0.3, label="ID Normal")
    axs[0].fill_between(bin_centers, 0, ood_hist, color='red', alpha=0.3, label="OOD Normal")
    axs[0].plot(bin_centers, id_1_hist, color='blue', alpha=0.6, linewidth=0.2)
    axs[0].plot(bin_centers, ood_hist, color='red', alpha=0.6, linewidth=0.2)

    # Plot IoU  
    iou, intersection, _ = compute_iou(id_1_hist, ood_hist)

    axs[0].fill_between(bin_centers, 0, intersection, color='darkgray', alpha=0.5, label=f"IoU = {iou:.4f}")
    axs[0].set_title("ID Normal vs OOD Normal (All)")

    axs[0].set_xlabel(score_function_name)
    axs[0].set_ylabel("Frequency")
    axs[0].legend()

    # Plot 2
    axs[1].fill_between(bin_centers, 0, id_2_hist, color='blue', alpha=0.3, label="ID Normal")
    axs[1].fill_between(bin_centers, 0, id_pertubated_hist, color='red', alpha=0.3, label="ID Pertubated")
    axs[1].plot(bin_centers, id_2_hist, color='blue', alpha=0.6, linewidth=0.2)
    axs[1].plot(bin_centers, id_pertubated_hist, color='red', alpha=0.6, linewidth=0.2)

    # Plot IoU
    iou, intersection, _ = compute_iou(id_2_hist, id_pertubated_hist)

    axs[1].fill_between(bin_centers, 0, intersection, color='darkgray', alpha=0.5, label=f"IoU = {iou:.4f}")

    axs[1].set_title("ID Normal vs ID Pertubated")
    axs[1].set_xlabel(score_function_name)

    # scale y axis to the same as the first plot
    axs[1].set_ylim(axs[0].get_ylim())
    axs[1].set_ylabel("Frequency")
    axs[1].legend()

    if show_title:
        fig.suptitle(f"{method_name_b}: {score_function_name} comparison for ID dataset {id_dataset_name}", fontsize=16)
        fig.subplots_adjust(top=0.85)

    # Save plot as pdf to results directory
    result_dir = get_top_level_path() + "/results/plots"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    plt.savefig(os.path.join(result_dir, f"{id_dataset}_perturbed_{method_name}_{score_function}_comparison.png"))
        
    plt.show()

def plot_baseline_comparison(id_dataset: str = "imagenet", ood_dataset: str = "", method_name: str = "", score_function: str = "energy_score", num_bins: int = 1000, show_title: bool = True) -> None:

    # Load the results
    baseline_result_dir = get_method_result_path(id_dataset)
    id_baseline_scores = np.load(os.path.join(baseline_result_dir, f"id_{id_dataset}.npy"), allow_pickle=True).item()[score_function]

    if ood_dataset:
        ood_baseline_scores = np.load(os.path.join(baseline_result_dir, f"ood_{ood_dataset}.npy"), allow_pickle=True).item()[score_function]
    else:
        # Append all OOD results in one np array
        ood_baseline_scores = []
        for ood_dataset_ in ['inat', 'dtd', 'sun', 'places']:
            ood_dataset_scores = np.load(os.path.join(baseline_result_dir, f"ood_{ood_dataset_}.npy"), allow_pickle=True).item()[score_function]
            ood_baseline_scores.extend(ood_dataset_scores)

    method_result_dir = get_method_result_path(id_dataset, method_name)

    id_method_scores = np.load(os.path.join(method_result_dir, f"id_{id_dataset}.npy"), allow_pickle=True).item()[score_function]

    if ood_dataset:
        ood_method_scores = np.load(os.path.join(method_result_dir, f"ood_{ood_dataset}.npy"), allow_pickle=True).item()[score_function]
    else:
        # Append all OOD results in one np array
        ood_method_scores = []
        for ood_dataset_ in ['inat', 'dtd', 'sun', 'places']:
            ood_dataset_scores = np.load(os.path.join(method_result_dir, f"ood_{ood_dataset_}.npy"), allow_pickle=True).item()[score_function]
            ood_method_scores.extend(ood_dataset_scores)

    # Beautify names
    id_dataset_name, ood_dataset_name, score_function_name, method_name_b = _beautify_names(id_dataset, ood_dataset, score_function, method_name)  

    # Make
    # Determine range of the histograms
    plot_bin_range_baseline = (
        min(min(id_baseline_scores), min(ood_baseline_scores)),
        max(max(id_baseline_scores), max(ood_baseline_scores))
    )
    plot_bin_range_method = (
        min(min(id_method_scores), min(ood_method_scores)),
        max(max(id_method_scores), max(ood_method_scores))
    )

    # Histograms for the first plot
    id_baseline_hist, bin_edges_baseline = np.histogram(id_baseline_scores, bins=num_bins, range=plot_bin_range_baseline, density=True)

    ood_baseline_hist, _ = np.histogram(ood_baseline_scores, bins=num_bins, range=plot_bin_range_baseline, density=True)

    # Histograms for the second plot
    id_method_hist, bin_edges_method = np.histogram(id_method_scores, bins=num_bins, range=plot_bin_range_method, density=True)

    ood_method_hist, _ = np.histogram(ood_method_scores, bins=num_bins, range=plot_bin_range_method, density=True)

    # Midpoints of bins for plotting
    bin_centers_baseline = (bin_edges_baseline[:-1] + bin_edges_baseline[1:]) / 2
    bin_centers_method = (bin_edges_method[:-1] + bin_edges_method[1:]) / 2 

    # Two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1
    axs[0].fill_between(bin_centers_baseline, 0, id_baseline_hist, color='blue', alpha=0.3, label="Baseline ID")
    axs[0].fill_between(bin_centers_baseline, 0, ood_baseline_hist, color='red', alpha=0.3, label="Baseline OOD")
    axs[0].plot(bin_centers_baseline, id_baseline_hist, color='blue', alpha=0.6, linewidth=0.2)
    axs[0].plot(bin_centers_baseline, ood_baseline_hist, color='red', alpha=0.6, linewidth=0.2)

    # Plot IoU
    iou, intersection, _ = compute_iou(id_baseline_hist, ood_baseline_hist)

    axs[0].fill_between(bin_centers_baseline, 0, intersection, color='darkgray', alpha=0.5, label=f"IoU = {iou:.4f}")

    if ood_dataset:
        axs[0].set_title(f"Baseline ID vs OOD {ood_dataset}")
    else:
        axs[0].set_title("Baseline ID vs OOD (All)")

    axs[0].set_xlabel(score_function_name)
    axs[0].set_ylabel("Frequency")
    axs[0].legend()

    # Plot 2
    axs[1].fill_between(bin_centers_method, 0, id_method_hist, color='blue', alpha=0.3, label=f"{method_name_b} ID")
    axs[1].fill_between(bin_centers_method, 0, ood_method_hist, color='red', alpha=0.3, label=f"{method_name_b} OOD")
    axs[1].plot(bin_centers_method, id_method_hist, color='blue', alpha=0.6, linewidth=0.2)
    axs[1].plot(bin_centers_method, ood_method_hist, color='red', alpha=0.6, linewidth=0.2)

    # Plot IoU
    iou, intersection, _ = compute_iou(id_method_hist, ood_method_hist)

    axs[1].fill_between(bin_centers_method, 0, intersection, color='darkgray', alpha=0.5, label=f"IoU = {iou:.4f}")

    axs[1].set_title(f"{method_name_b} ID vs OOD")
    axs[1].set_xlabel(score_function_name)
    # scale y axis to the same as the first plot
    #axs[1].set_ylim(axs[0].get_ylim())
    axs[1].set_ylabel("Frequency")
    axs[1].legend()

    if show_title:
        fig.suptitle(f"{method_name_b}: {score_function_name} comparison for ID dataset {id_dataset_name}", fontsize=16)
        fig.subplots_adjust(top=0.85)

    # Save plot as pdf to results directory
    result_dir = get_top_level_path() + "/results/plots"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    plt.savefig(os.path.join(result_dir, f"{id_dataset}_{method_name}_{score_function}_baseline_comparison.pdf"))

    plt.show()

def _beautify_names(id_dataset_name: str = "imagenet", ood_dataset_name: str = "", score_function: str = "energy_score", method_name: str = "") -> tuple[str, str, str, str]:
    if id_dataset_name == "imagenet":
        id_dataset_name = "ImageNet"

    if ood_dataset_name:
        if ood_dataset_name == "inat":
            ood_dataset_name = "iNaturalist"
        elif ood_dataset_name == "dtd":
            ood_dataset_name = "DTD"
        elif ood_dataset_name == "sun":
            ood_dataset_name = "SUN"
        elif ood_dataset_name == "places":
            ood_dataset_name = "Places365"

    if score_function == "energy_score":
        score_function = "Energy Score"
    elif score_function == "msp_score":
        score_function = "MSP Score"
    elif score_function == "entropy_score":
        score_function = "Entropy Score"
    else:
        score_function = "Confidence"

    if method_name == "baseline":
        method_name = "Baseline"
    elif method_name == "react":
        method_name = "ReAct"
    elif method_name == "bats":
        method_name = "BATS"
    elif method_name == "ashs":
        method_name = "AshS"
    elif method_name == "ashb":
        method_name = "AshB"
    elif method_name == "ashp":
        method_name = "AshP"
    elif method_name == "dice":
        method_name = "Dice"
    elif method_name == "line":
        method_name = "Line"
    elif method_name == "react_ashs":
        method_name = "ReAct + AshS"
    elif method_name == "react_ashb":
        method_name = "ReAct + AshB"
    elif method_name == "react_ashp":
        method_name = "ReAct + AshP"
    elif method_name == "react_dice":
        method_name = "ReAct + Dice"
    elif method_name == "react_line":
        method_name = "ReAct + Line"

    return id_dataset_name, ood_dataset_name, score_function, method_name

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