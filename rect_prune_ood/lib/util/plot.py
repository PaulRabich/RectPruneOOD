import numpy as np
import matplotlib.pyplot as plt

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
    data_1_color = 'lightblue'
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