from typing import List, Literal
import torch
from torch import Tensor
from torch.nn import Module

mask_types = Literal["none","bool", "scale"]

class LayerScaler(torch.nn.Module):
    def __init__(self, mask_type:mask_types, indeces:List[int], num_dim:int) -> None:
        #super(LayerScaler, self).__init__()
        super().__init__()  
        self.construct_mask("bool", indeces, num_dim, {})
        
    def construct_mask(self, type:mask_types, indeces:List[int], num_dim:int, kwargs:dict) -> Tensor:
        if type == "none":
            mask = torch.ones(num_dim)
            
        elif type == "bool":
            mask = torch.zeros(num_dim)
            mask[indeces] = 1
        
        elif type == "scale":
            in_scale = kwargs["in_scale"]
            out_scale = kwargs["out_scale"]

            mask = torch.ones(num_dim) * in_scale
            mask[indeces] = out_scale

        else:
            raise ValueError("Invalid scaling type")
        
        self.mask = mask.cuda()

        return mask
    def forward(self, x: Tensor) -> Tensor:
        x = x * self.mask
        return x
    


