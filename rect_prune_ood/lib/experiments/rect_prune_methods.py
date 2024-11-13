from collections import defaultdict
import torch
import numpy as np
import torch.nn as nn
import os


#methods to use from run experiment:

#method if nothing is chosen, modifies nothing

class BasePruningModule(torch.nn.Module):
    def __init__(self, fc_layer):
        super(BasePruningModule, self).__init__()
        # Enforcing that all subclasses must set `self.fc_layer`
        self.fc_layer = fc_layer


class NeutralPruningModule(BasePruningModule):
    def __init__(self, fc_layer):
        super(NeutralPruningModule, self).__init__(fc_layer)

    def forward(self, x):
        #x = torch.flatten(x, 1)
        return self.fc_layer(x)
    



class AshBPruningModule(BasePruningModule):
    def __init__(self, fc_layer, percentage=90):
        super(AshBPruningModule, self).__init__(fc_layer)
        self.percentage = percentage

    def forward(self, x):
        x = self.ash_b(x, self.percentage)
        #x = torch.flatten(x, 1)
        return self.fc_layer(x)
    
    def ash_b(x, percentile=65):
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
    
class AshSPruningModule(BasePruningModule):
    def __init__(self, fc_layer, percentage=90):
        super(AshSPruningModule, self).__init__(fc_layer)
        self.percentage = percentage

    def forward(self, x):
        x = self.ash_s(x, self.percentage)
        #x = torch.flatten(x, 1)
        return self.fc_layer(x)
    
    def ash_s(x, percentile=65):
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
    
class AshPPruningModule(BasePruningModule):
    def __init__(self, fc_layer, percentage=90):
        super(AshPPruningModule, self).__init__(fc_layer)
        self.percentage = percentage

    def forward(self, x):
        x = self.ash_p(x, self.percentage)
        #x = torch.flatten(x, 1)
        return self.fc_layer(x)
    
    def ash_p(x, percentile=65):
        assert x.dim() == 4
        assert 0 <= percentile <= 100

        b, c, h, w = x.shape

        n = x.shape[1:].numel()
        k = n - int(np.round(n * percentile / 100.0))
        t = x.view((b, c * h * w))
        v, i = torch.topk(t, k, dim=1)
        t.zero_().scatter_(dim=1, index=i, src=v)

        return x
    

class DicePruningModule(BasePruningModule):
    def __init__(self, fc_layer, p=90, num_classes=100, info_path=os.path.join("RectPruneOOD", "cache", "CIFAR-10_densenet_feat_stat.npy")):
        self.p = p
        self.info_path = info_path

        info = np.load(self.info_path) #f"cache/{args.in_dataset}_{args.model_arch}_feat_stat.npy")
        in_features = 512 * 4 #block.expansion
        out_features = num_classes
        last_layer = RouteDICE(in_features, out_features, p=p, info=info)

        super(DicePruningModule, self).__init__(last_layer)

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.fc_layer(x)
    
    

class NeutralRectModule(torch.nn.Module):
    def __init__(self):
        super(NeutralRectModule, self).__init__()

    def forward(self, x):
        return x



class ReactRectModule(torch.nn.Module):
    def __init__(self, threshold=1.0):
        super(ReactRectModule, self).__init__()
        self.threshold = threshold

    def forward(self, x):
        return x.clip(max=self.threshold)
    

class BatsRectModule(torch.nn.Module):
    def __init__(self, dataset:str = "ImageNet", mean_path: str = os.path.join("RectPruneOOD", "cache", "vit_features_std.pt"), std_path:str = os.path.join("RectPruneOOD", "cache", "vit_features_mean.pt")):
        super(ReactRectModule, self).__init__()
        self.mean_path = mean_path
        self.std_path = std_path
        self.dataset = dataset

    def forward(self, x):
        #mean std data
        feature_std=torch.load(self.mean_path).cuda()
        feature_mean=torch.load(self.std_path).cuda() 

        if self.dataset == 'ImageNet':
            lam = 1.05
        elif self.dataset == 'CIFAR':
            lam = 3.25
        else:
            raise Exception("Unsupported dataset")

        x = torch.where(x<(feature_std*lam+feature_mean),x,feature_std*lam+feature_mean)
        x = torch.where(x>(-feature_std*lam+feature_mean),x,-feature_std*lam+feature_mean)
        return x

#ash, distinguishes between three ash modes
def pruning_ash(x, percentile=65, mode="p"):
    if mode == "p":
         return ash_p(x, percentile)
    if mode == "s":
         return ash_s(x, percentile)
    if mode == "b":
         return ash_b(x, percentile)
    else:
        raise Exception("Unsupported ashe mode provided")
    #match mode:
    #    case "p":
    #        return ash_p(x, percentile)
    #    case "s":
    #        return ash_s(x, percentile)
    #    case "b":
    #        return ash_b(x, percentile)
    #    case _:
    #        raise Exception("Unsupported ashe mode provided")
    

#dice, distinguishes between three ash modes
def pruning_dice(x):
    
    p = 90
    weight_info = "CIFAR-10_densenet_feat_stat.npy"
    info = np.load(os.path.join("RectPruneOOD", "cache", weight_info)) #f"cache/{args.in_dataset}_{args.model_arch}_feat_stat.npy")
    in_features = 512 * 4 #block.expansion
    out_features = 100 #num_classes
    last_layer = RouteDICE(in_features, out_features, p=p, info=info)

    return x #, last_layer
    


pruning_methods = {"ash": pruning_ash, "dice": pruning_dice}
rectification_method = {"react": rectification_react, "bats": rectification_bats}

 
    


pruning_classes = defaultdict(lambda: NeutralPruningModule, {"ashp": AshPPruningModule, "ashb": AshBPruningModule, "ashs": AshSPruningModule, "dice": DicePruningModule})
rectification_classes = defaultdict(lambda: NeutralRectModule, {"react": ReactRectModule, "bats": BatsRectModule})


def get_methods(rectification, pruning):
    recting = rectification_method[rectification]
    pruning = pruning_methods[pruning]
def get_classes(rectification, pruning):
    rectification_class = rectification_classes[rectification]
    pruning_class = pruning_classes[pruning]

    return rectification_class, pruning_class


#gejoinkt direkt aus ash repo
def ash_b(x, percentile=65):
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


def ash_p(x, percentile=65):
    assert x.dim() == 4
    assert 0 <= percentile <= 100

    b, c, h, w = x.shape

    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    return x


def ash_s(x, percentile=65):
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



#fully connected layer that is weightpruned
class RouteDICE(nn.Linear):
                                                            #p=70 für imagenet in .sh
    def __init__(self, in_features, out_features, bias=True, p=90, conv1x1=False, info=None):
        super(RouteDICE, self).__init__(in_features, out_features, bias)
        if conv1x1:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features, 1, 1))
        self.p = p
        self.info = info
        self.masked_w = None

    def calculate_mask_weight(self):
        self.contrib = self.info[None, :] * self.weight.data.cpu().numpy()
        # self.contrib = np.abs(self.contrib)
        # self.contrib = np.random.rand(*self.contrib.shape)
        # self.contrib = self.info[None, :]
        # self.contrib = np.random.rand(*self.info[None, :].shape)
        self.thresh = np.percentile(self.contrib, self.p)
        mask = torch.Tensor((self.contrib > self.thresh))
        self.masked_w = (self.weight.squeeze().cpu() * mask).cuda()

    def forward(self, input):
        if self.masked_w is None:
            self.calculate_mask_weight()
        vote = input[:, None, :] * self.masked_w.cuda()
        if self.bias is not None:
            out = vote.sum(2) + self.bias
        else:
            out = vote.sum(2)
        return out






