import torch
from torch.nn import Module


class Pipeline(Module):
    def __init__(self, featureExtractor, rect_method, prune_method) -> None:
        super().__init__()
        self.featureExtractor = featureExtractor
        self.rect_method = rect_method
        self.prune_method = prune_method

    def forward(self, x) -> torch.Tensor:
        out = self.featureExtractor(x)
        rectified_out = self.rect_method(out)
        pruned_out = self.prune_method(rectified_out)

        return pruned_out






