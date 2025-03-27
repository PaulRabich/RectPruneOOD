from typing import List, Literal
import os
import torch
from torch import Tensor
from torch.nn import Module
import numpy as np

from rect_prune_ood.lib.util.paths import get_top_level_path

mask_types = Literal["none", "bool", "scale"]

def imagewise_variance(perturbed_activations, p=90, num_augments=4):
    assert perturbed_activations.shape[0] % num_augments == 0
    #group perturbed activations by image
    all_activations = perturbed_activations.reshape(int(perturbed_activations.shape[0]/num_augments), num_augments, perturbed_activations.shape[1])
    #take mean of imagewise activations to get rough activation level on perturbed images in general
    #maybe this step can be skipped for experimentiation
    #mean_imagewise_perturbed = grouped_perturbed.mean(axis=1)

    #concatenate perturbed and normal acticvations
    #all_activations = np.stack((normal_activations, grouped_perturbed), axis=1)

    iamgewise_variance = np.var(all_activations, axis=1)

    max_indices = np.argmax(iamgewise_variance, axis=1)

    index_counts = np.bincount(max_indices, minlength=2048)
    ranked_indices = np.argsort(index_counts)[::-1]

    top_10th_count = int(len(ranked_indices) * (1-p/100))
    ranked_indices = ranked_indices[top_10th_count:]

    return ranked_indices

def construct_mask(type:mask_types, percentile:int, kwargs:dict, imagewise=False) -> Tensor:
        if imagewise:
            cache_path = os.path.join(get_top_level_path(), "results", "imagenet", "baseline_mobilenet", "imagenet_perturbed_activations.npy")
        else:
            cache_path = os.path.join(get_top_level_path(), "cache", "imagenet_perturbed_variance_mobilenet.npy")
        
        variances = np.load(cache_path, allow_pickle="True")
        
        if imagewise:
            num_dim = variances.shape[1]
        else:
            num_dim = variances.shape[0]

        if imagewise:
            top_k_percentile_indeces = imagewise_variance(variances, percentile)
        else:
            k_percentile = np.percentile(variances, percentile)
            top_k_percentile_indeces = np.where(variances <= k_percentile)[0]
        
        if type == "none":
            mask = torch.ones(num_dim)
            
        elif type == "bool":
            mask = torch.zeros(num_dim)
            mask[top_k_percentile_indeces.copy()] = 1
        
        elif type == "scale":
            in_scale = kwargs["in_scale"]
            out_scale = kwargs["out_scale"]

            mask = torch.ones(num_dim) * in_scale
            mask[top_k_percentile_indeces.copy()] = out_scale

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
    def __init__(self, percentile) -> None:
        #super(LayerScaler, self).__init__()
        super(APESB, self).__init__()
        self.percentile = percentile
        self.mask = construct_mask("bool", percentile, {})
        
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.apes_b(x)
        return x
    

    def apes_b(self, x):
        assert x.dim() == 4
        b, c, h, w = x.shape
        assert self.mask.shape[0] == c 

        # calculate the sum of the input per sample
        s1 = x.sum(dim=[1, 2, 3])

        mask_sum = self.mask.sum() * h * w
        active_elements = mask_sum.item()
        fill = s1 / active_elements
        expanded_fill = fill.view(b, 1, 1, 1)
        expanded_mask = self.mask.expand((b, *self.mask.shape))
        x_out = expanded_fill * expanded_mask
       
        assert x_out.shape == (b, c, h, w) #in case i didnt think correctly
        return x_out
    
    def __str__(self):
        return super().__str__() + f" with percentile {self.percentile}"
    
class APESP(Module):
    def __init__(self, percentile) -> None:
        #super(LayerScaler, self).__init__()
        super(APESP, self).__init__()
        self.percentile = percentile
        self.mask = construct_mask("bool", percentile, {})
        
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.apes_p(x)
        return x
    
    def apes_p(self, x):
        assert x.dim() == 4
        b, c, h, w = x.shape
        assert self.mask.shape[0] == c 

        x = x * self.mask

        return x

    def __str__(self):
        return super().__str__() + f" with percentile {self.percentile}"

class APESS(Module):
    def __init__(self, percentile) -> None:
        #super(LayerScaler, self).__init__()
        super(APESS, self).__init__()
        self.percentile = percentile
        self.mask = construct_mask("bool", percentile, {})
        
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.apes_s(x)
        #x = x * self.mask
        return x

    def apes_s(self, x):
        assert x.dim() == 4
        b, c, h, w = x.shape
        assert self.mask.shape[0] == c

        # calculate the sum of the input per sample
        s1 = x.sum(dim=[1, 2, 3])
      
        x = x * self.mask
        # calculate new sum of the input per sample after pruning
        s2 = x.sum(dim=[1, 2, 3])

        # apply sharpening
        scale = s1 / s2
        x = x * torch.exp(scale[:, None, None, None])

        return x

    def __str__(self):
        return super().__str__() + f" with percentile {self.percentile}"
    


class APESSCALE(Module):
    def __init__(self, percentile) -> None:
        #super(LayerScaler, self).__init__()
        super(APESSCALE, self).__init__()
        self.percentile = percentile
        self.mask = construct_mask("bool", percentile, {})
        
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.apes_scale(x)
        #x = x * self.mask
        return x

    def apes_scale(self, x):
        input = x.clone()
        assert x.dim() == 4
        
        b, c, h, w = x.shape

        # calculate the sum of the input per sample
        s1 = x.sum(dim=[1, 2, 3])
        
        x = x * self.mask

        # calculate new sum of the input per sample after pruning
        s2 = x.sum(dim=[1, 2, 3])

        # apply sharpening
        scale = s1 / s2
        #apply scaling to unchanged input
        return input * torch.exp(scale[:, None, None, None])

    def __str__(self):
        return super().__str__() + f" with percentile {self.percentile}"
