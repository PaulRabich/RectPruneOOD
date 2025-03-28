import sys
import os
import numpy as np

module_path = os.path.abspath(os.path.join('..', ''))
if module_path not in sys.path:
    sys.path.append(module_path)

from rect_prune_ood.lib.util.paths import get_top_level_path

#how to generate the variance used in our mask calculation for APES
#you need: 
#   - the resulting .npy files from running our pipeline on the "OOD" dataset imagenet_perturbed with caputure_activations=True (will be saved in results dir)
#   - activations are specific to backbone used 


#path to results
results_dir = os.path.join(get_top_level_path(), 'results', 'imagenet', "baseline_no_pruning")

#load activations from file
activations_perturbed = np.load(os.path.join(results_dir, "imagenet_perturbed_activations.npy"), allow_pickle=True)


#calculate variance
variance = np.std(activations_perturbed, axis=0)

#this variance or similar variance is used in construct_mask in apes.py 








