from collections import defaultdict
import torch
import numpy as np
import torch.nn as nn
import os


#methods to use from run experiment:

#method if nothing is chosen, modifies nothing
def neutral_function(x):
    return x

#react
def rectification_react(x, threshold=1.0):
    #og line from repo
    return x.clip(max=threshold)


def rectification_bats(x):
    #std data
    dataset = "ImageNet"
    if dataset == 'ImageNet':
        feature_std=torch.load(os.path.join("RectPruneOOD", "cache", "vit_features_std.pt")).cuda()
        feature_mean=torch.load(os.path.join("RectPruneOOD", "cache", "vit_features_mean.pt")).cuda()        
        lam = 1.05
    elif dataset == 'CIFAR':
        feature_std=torch.load(os.path.join("RectPruneOOD", "cache", "cifar_features_std.pt")).cuda()
        feature_mean=torch.load(os.path.join("RectPruneOOD", "cache", "cifar_features_mean.pt")).cuda()
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


def get_methods(rectification, pruning):
    recting = rectification_method[rectification]
    pruning = pruning_methods[pruning]

    return recting, pruning


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






