# Pruning-based OOD Detection via Activation Consistency Under Extraneous Shifts
# Installation
To be able to use the project and to test different masks run:
```shell 
pip install -r requirements.txt
pip install .
```

# Scripts and Functionality
Out of the box this repository provide the following scripts:
1. `create_mask.py` This scipt is for creating new masks and making them available for the APES method. To create a mask provide a `name` that will be used later for using the mask. *It is important that the name does not contain an underscore '_'*. Also provide a list of perturbations that should be used for the mask. A full list of perturbations can be found [here](#perturbations).
```shell 
python create_mask.py --backbone resnet50 --percentile 95 --name mask-name --severity 1 --values 1 4 5 7 12
```
2. `calculate_score.py` Ths script runs the OOD detectio process. As ID dataset is ImageNet1k used. The parameter `method_names` takes a list of method names to be ran with the script. If one wishes to run the apes method, refer to the preferred mask with `apes@mask-name`. The second parameter `datasets` takes a list of the available datasets to be used as a OOD dataset. 
```shell
python calculate_score.py --method_name apes@sev1-all --datasets inat
```
3. `calculate_full_spec_score.py` This script runs Full Spectrum Experiment with the provided method names and dataset. The first parameter is `method_names`, which specifies the method to be used. The second parameter is `datasets` descibing the datasets that are run as OOD datasets. This experiment compute the score of ImagetNet + ImageNet-C againset the OOD dataset.
```shell 
python calculate_full_spectrum_scores.py --method_name ashs_react --datasets inat
```
4. `ablation_percentiles.py` This script runs the ablation study presented in the main report. It also provides some parameters to change the ablation study. `percentiles` is a list with the percentiles that should be used for the study.
```shell 
python ablation_percentiles.py --severity 1 --percentiles 5 10 55 90 95
```

# Method Naming
To use a specific method pass the method name in lowercase as the `method_name` parameter. Examples are `"react" or "ashs"`. For Ash and APES you can also provide a parameter for the method to be used, append this parameter with a "@" to the name: `"ashs@65", apes@top-mask"`. The Ash parameter describes the percentile used for the method. The APES parameter specifies the mask to be used for the method. Make sure to create the mask first with the `crate_mask.py` script.

# Datasets
This project provides the following datasets:
1. Imagenet1k (`imagenet`)
2. ImageNet-C (`imagenet-c`)
3. OpenImage-O (`openiamge-o`)
4. iNaturalist (`inat`)
5. SUN (`sun`)
6. Places (`places`)
7. Textures (`dtd`)
8. Ninco (`ninco`)
9. SSB Hard (`ssb-hard`)

### Download
In order to download the datasets, we provide a modified version of the script of the [official OpenOOD benchmark repository](https://github.com/Jingkang50/OpenOOD). The script is located in `rect_prune_ood/scripts/` and can simply be run with:
```shell
python download_datasets.py 
```
This will download all the necessary datasets to `RectPruneOOD/datasets/`. 

# Perturbations
The perturbations that can be used for creating the mask are:
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

Note that this are the 19 ImageNet-C perturbations. When creating masks with the `create_mask.py` script refer to the indecies from the list above.

