from typing import List, Literal
import os
import torch
from torch import Tensor
from torch.nn import Module
import numpy as np

from rect_prune_ood.lib.util.paths import get_top_level_path

mask_types = Literal["none", "bool", "scale"]

def construct_mask(type:mask_types, percentile:int, kwargs:dict) -> Tensor:

        cache_path = os.path.join(get_top_level_path(), "cache")
        variances = np.load(os.path.join(cache_path, "imagenet_variance.npy"))
        num_dim = variances.shape[0]

        k_percentile = np.percentile(variances, percentile)
        # > or <= ????
        top_k_percentile_indeces = np.where(variances <= k_percentile)[0]

        if type == "none":
            mask = torch.ones(num_dim)
            
        elif type == "bool":
            mask = torch.zeros(num_dim)
            mask[top_k_percentile_indeces] = 1
        
        elif type == "scale":
            in_scale = kwargs["in_scale"]
            out_scale = kwargs["out_scale"]

            mask = torch.ones(num_dim) * in_scale
            mask[top_k_percentile_indeces] = out_scale

        else:
            raise ValueError("Invalid scaling type")
        
        mask = mask.view(num_dim, 1, 1)
        mask = mask.cuda()

        return mask

class APES(Module):
    def __init__(self, percentile) -> None:
        #super(LayerScaler, self).__init__()
        super().__init__()
        self.percentile = percentile
        self.mask = construct_mask("bool", percentile, {})
        
    
    def forward(self, x: Tensor) -> Tensor:
        x = x * self.mask
        return x
    
class APESB(Module):
    def __init__(self, percentile, in_scale, out_scale) -> None:
        #super(LayerScaler, self).__init__()
        super().__init__()
        self.percentile = percentile
        construct_mask("bool", {})
        
    
    def forward(self, x: Tensor) -> Tensor:
        x = x * self.mask
        return x
    
