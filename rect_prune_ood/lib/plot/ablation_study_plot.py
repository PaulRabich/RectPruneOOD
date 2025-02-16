import matplotlib.pyplot as plt
from rect_prune_ood.lib.eval.metrics import compute_traditional_ood

def plot_ablation_study(id_dataset="imagenet", ood_datasets=["openimage-o"], method_name="apesp"):

    fpr = []
    auroc = []
    percentiles =  range(35, 105, 5)

    for percentile in percentiles:
        name = f"{method_name}@{percentile}"
        metrics = compute_traditional_ood(
            id_dataset=id_dataset,
            ood_datasets=ood_datasets,
            method_name=name,
            print_results=False
        )

        fpr.append(metrics[0]["FPR"])
        auroc.append(metrics[0]["AUROC"])
    


    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    axs[0].plot(percentiles, fpr, label='FPR (↓)', marker='o', linestyle='dotted', color='orange')
    axs[0].set_xticks(percentiles)
    axs[0].set_xlabel('Percentile', fontsize=12)
    axs[0].set_ylabel('Value (%)', fontsize=12)
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(percentiles, auroc, label='AUROC (↑)', marker='o', linestyle='dotted')
    axs[1].set_xticks(percentiles)
    axs[1].set_xlabel('Percentile', fontsize=12)
    # axs[1].set_ylabel('Value (%)', fontsize=12)
    axs[1].legend()
    axs[1].grid()