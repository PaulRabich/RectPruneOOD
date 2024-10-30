from typing import Any
import torch
from torch.nn import Module


class React(Module):
    def __init__(self, clip_threshold:float) -> None:
        super().__init__()
        self.clip_threshold = clip_threshold


    def forward(self, x:Any):
        # apply react
        preultimate_react = torch.clip(input=x, min=0, max=self.clip_threshold)
        return preultimate_react


