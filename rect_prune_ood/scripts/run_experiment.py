import sys
sys.path.append("./")

import argparse

from rect_prune_ood.lib.experiments.run_experiment import run_experiment

# Define the parser and set up all the arguments
#parser = argparse.ArgumentParser(description="Define the experiement to run. Changable are the different pruning and rectification methods, the dataset and the backbone.")
#parser.add_argument("--backbone", type=str, help="Backbone to use")
#parser.add_argument("--layer", type=str, help="Layer to use")
#parser.add_argument("--dataset", type=str, help="Dataset to use")
#parser.add_argument("--pruning", type=str, help="Pruning method to use")
#parser.add_argument("--rectification", type=str, help="Rectification method to use")

# Parse command-line arguments
#args = parser.parse_args()

# Call the search function with the provided arguments
# run_experiment(args.backbone, args.layer, args.dataset, args.pruning, args.rectification)
run_experiment(backbone="resnet50", layer="avgpool", dataset="imagenet", rectification_method="", prune_method="")