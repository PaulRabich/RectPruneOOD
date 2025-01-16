# Changelog
## Tim: 16.01.2025
Rework "`id_dataset`" specification.
### Additions
- [load_scores_for_method()](./rect_prune_ood/lib/util/results_loader.py): Load scores for all datasets for a given method in a dictionary.
### General Changes
- `id_dataset` -> `base_dataset` in all methods that used `id_dataset` for the path. 
- For [compute_traditional_ood()](./rect_prune_ood/lib/eval/metrics.py) the keyword `id_dataset` now works as intended.