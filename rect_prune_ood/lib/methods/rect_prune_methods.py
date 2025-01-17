from collections import defaultdict
import torch
import numpy as np
from torch.nn import Module
import os
from rect_prune_ood.lib.methods.route_classes import RouteDICE, RouteLUNCH
from rect_prune_ood.lib.methods.route_classes import Bottleneck as BN



class AshBPruningModule(Module):
    def __init__(self, percentage=65):
        super(AshBPruningModule, self).__init__()
        self.percentage = percentage

    def forward(self, x):
        x = self.ash_b(x, self.percentage)
        return x

    def ash_b(self, x, percentile=65):
        assert x.dim() == 4
        assert 0 <= percentile <= 100
        b, c, h, w = x.shape

        # calculate the sum of the input per sample
        s1 = x.sum(dim=[1, 2, 3])

        n = x.shape[1:].numel()
        k = n - int(np.round(n * percentile / 100.0))
        t = x.view((b, c * h * w))
        v, i = torch.topk(t, k, dim=1)
        fill = s1 / k
        fill = fill.unsqueeze(dim=1).expand(v.shape)
        t.zero_().scatter_(dim=1, index=i, src=fill)
        return x

    def __str__(self):
        return super().__str__() + f" with percentage {self.percentage}"

class AshSPruningModule(Module):
    def __init__(self, percentage=90):
        super(AshSPruningModule, self).__init__()
        self.percentage = percentage

    def forward(self, x):
        x = self.ash_s(x, self.percentage)
        return x

    def ash_s(self, x, percentile=65):
        assert x.dim() == 4
        assert 0 <= percentile <= 100
        b, c, h, w = x.shape

        # calculate the sum of the input per sample
        s1 = x.sum(dim=[1, 2, 3])
        n = x.shape[1:].numel()
        k = n - int(np.round(n * percentile / 100.0))
        t = x.view((b, c * h * w))
        v, i = torch.topk(t, k, dim=1)
        t.zero_().scatter_(dim=1, index=i, src=v)

        # calculate new sum of the input per sample after pruning
        s2 = x.sum(dim=[1, 2, 3])

        # apply sharpening
        scale = s1 / s2
        x = x * torch.exp(scale[:, None, None, None])

        return x

    def __str__(self):
        return super().__str__() + f" with percentage {self.percentage}"

class AshPPruningModule(Module):
    def __init__(self, percentage=50):
        super(AshPPruningModule, self).__init__()
        self.percentage = percentage

    def forward(self, x):
        x = self.ash_p(x, self.percentage)
        return x

    def ash_p(self, x, percentile=50):
        assert x.dim() == 4
        assert 0 <= percentile <= 100

        b, c, h, w = x.shape

        n = x.shape[1:].numel()
        k = n - int(np.round(n * percentile / 100.0))
        t = x.view((b, c * h * w))
        v, i = torch.topk(t, k, dim=1)
        t.zero_().scatter_(dim=1, index=i, src=v)

        return x
    
    def __str__(self):
        return super().__str__() + f" with percentage {self.percentage}"


class DicePruningModule(Module):
    def __init__(self, fc_layer, p=90, num_classes=1000, info_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "imagenet_resnet50_feat_stat.npy"), clip_threshold=1e5):
        self.p = p
        self.info_path = info_path

        # f"cache/{args.in_dataset}_{args.model_arch}_feat_stat.npy")
        info = np.load(self.info_path)
        in_features = 512 * 4  # block.expansion
        out_features = num_classes
        last_layer = RouteDICE(in_features, out_features, p=p, info=info_path)
        self.clip_threshold = clip_threshold

        super(DicePruningModule, self).__init__(last_layer)

    def forward(self, x):
        raise NotImplementedError("This method should not be called. Problem with the fully connected layer at the end.")
        #x = x.clip(max=self.clip_threshold)
        x = x.view(x.size(0), -1)
        return self.fc_layer(x)



class ReactRectModule(Module):
    def __init__(self, threshold=1.0):
        super(ReactRectModule, self).__init__()
        self.threshold = threshold

    def forward(self, x):
        return x.clip(max=self.threshold)
    
    def __str__(self):
        return super().__str__() + f" with threshold {self.threshold}"

class BatsRectModule(torch.nn.Module):
    def __init__(self, dataset: str = "ImageNet", mean_path: str = os.path.join("RectPruneOOD", "cache", "vit_features_std.pt"), std_path: str = os.path.join("RectPruneOOD", "cache", "vit_features_mean.pt")):
        super().__init__()
        self.mean_path = mean_path
        self.std_path = std_path
        self.dataset = dataset
        #if self.dataset == 'ImageNet':
        # mean std data
        self.feature_std = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_batchnorm_mean.pt")).cuda()
        self.feature_mean = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_batchnorm_std.pt")).cuda()
        self.lam = 1.05


    def forward(self, x):
        # mean std data
        #feature_std = torch.load(self.mean_path).cuda()
        #feature_mean = torch.load(self.std_path).cuda()

        x = x.view(x.size(0), -1)
        x = torch.where(x < (self.feature_std*self.lam+self.feature_mean),
                        x, self.feature_std*self.lam+self.feature_mean)
        x = torch.where(x > (-self.feature_std*self.lam+self.feature_mean),
                        x, -self.feature_std*self.lam+self.feature_mean)
        return x


