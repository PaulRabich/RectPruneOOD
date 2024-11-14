import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_msp_score(inputs, model,  method_args, logits=None):
    scores = np.max(F.softmax(logits, dim=1).detach().cpu().numpy(), axis=1)
    return scores

def get_energy_score(inputs, model, method_args, logits=None):
    scores = torch.logsumexp(logits.data.cpu(), dim=1).numpy()
    return scores

def get_odin_score(inputs, model, forward_func, method_args):
    temper = method_args['temperature']
    noiseMagnitude1 = method_args['magnitude']

    criterion = nn.CrossEntropyLoss()
    inputs = torch.autograd.Variable(inputs, requires_grad = True)
    # outputs = model(inputs)
    outputs = forward_func(inputs, model)

    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

    # Using temperature scaling
    outputs = outputs / temper

    labels = torch.autograd.Variable(torch.LongTensor(maxIndexTemp).cuda())
    loss = criterion(outputs, labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient =  torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
    # outputs = model(Variable(tempInputs))
    with torch.no_grad():
        outputs = forward_func(tempInputs, model)
    outputs = outputs / temper
    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
    scores = np.max(nnOutputs, axis=1)

    return scores


def get_score(inputs, model, method, method_args = None, logits=None):
    if method == "msp":
        scores = get_msp_score(inputs, model,  method_args, logits)
    elif method == "odin":
        scores = get_odin_score(inputs, model,  method_args)
    elif method == "energy":
        scores = get_energy_score(inputs, model, method_args, logits)
    return scores