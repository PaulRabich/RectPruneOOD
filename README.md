# Pruning-based OOD Detection via Activation Consistency Under Extraneous Shifts



## Datasets
---
This directory contains the datasets used in the experiments. The datasets are stored in the `datasets` directory. The datasets are stored in the following format.
```
datasets
├── id_datasets
│   ├── cifar10
│   │   ├── cifar-10-batches-py
├── ood_datasets
│   ├── dtd
```
The subdirectories are named `id_datasets` and `ood_datasets` for in-distribution and out-of-distribution datasets, respectively.



## Code Functionalities Fragen.
- visualize_distrbutions: 
- test_experiment:
- was macht scripts/calculate_full_spectrum?
- was macht scripts/calculate_score?

## Kommentiert:
- scripts/calculate_imagenet_variance
- lib/eval/.
- lib/experiments

## Kann weg?:
Notebooks:
- base_notebook
- capture_bats
- test_image_perturbations
- train_cifar_resnet

Scripts:
- percentile_ablation_study: wurde ja durch tims branche ersetz oder?
- train_resnet_cifar

Lib:
- models/resnet_cifar_react + der ganze ordner dann
- der ganze precomputation ordner?

# Perturbations
1. Brightness  
2. Defocus Blur  
3. Fog  
4. Gaussian Blur  
5. Glass Blur  
6. JPEG Compression  
7. Pixelate  
8. Shot Noise  
9. Spatter  
10. Zoom Blur  
11. Contrast  
12. Elastic Transform  
13. Frost  
14. Gaussian Noise  
15. Impulse Noise  
16. Motion Blur  
17. Saturate  
18. Snow  
19. Speckle Noise  
