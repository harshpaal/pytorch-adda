"""Utilities for ADDA."""

import os
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.models as models

import params
from datasets import get_mnist, get_usps

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_variable(tensor, volatile=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, volatile=volatile)


def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def denormalize(x, std, mean):
    """Invert normalization, and then convert array into image."""
    out = x * std + mean
    return out.clamp(0, 1)


def init_weights(layer):
    """Init weights for layers w.r.t. the original paper."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif layer_name.find("BatchNorm") != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)


def init_random_seed(manual_seed):
    """Init random seed."""
    seed = None
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_data_loader(name, train=True):
    """Get data loader by name."""
    if name == "MNIST":
        return get_mnist(train)
    elif name == "USPS":
        return get_usps(train)


class fully_connected(nn.Module):
    """docstring for BottleNeck"""
    def __init__(self, model, num_ftrs, num_classes):
        super(fully_connected, self).__init__()
        self.model = model
        self.fc_4 = nn.Linear(num_ftrs,num_classes)

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)
        out_1 = x
        out_3 = self.fc_4(x)
        return  out_1, out_3

#loading KimiaNet
def load_kimiaNet(pt_model_path, input_size, num_classes):
    model = models.densenet121(pretrained=True)
    for param in model.parameters():
      param.requires_grad = False

    model.features = nn.Sequential(model.features , nn.AdaptiveAvgPool2d(output_size= (1,1)))
    num_ftrs = model.classifier.in_features
    model_final = fully_connected(model.features, num_ftrs, 30)
    model = model.to(device)
    model_final = model_final.to(device)
    model_final = nn.DataParallel(model_final)
    params_to_update = []
    criterion = nn.CrossEntropyLoss()

    model_final.load_state_dict(torch.load(pt_model_path,
                                          map_location=torch.device(device)))
    model_final.module.fc_4 = nn.Linear(1024, num_classes)
    print("Params to learn:")
    params_to_update = []
    for name,param in model_final.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)

    # return encoder, classifier
    return model_final.module.model, model_final.module.fc_4




def init_model(net, restore):
    """Init models with cuda and weights."""
    # init weights of model
    net.apply(init_weights)

    # restore model weights
    if restore is not None and os.path.exists(restore):
        net.load_state_dict(torch.load(restore))
        net.restored = True
        print("Restore model from: {}".format(os.path.abspath(restore)))

    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()

    return net


def save_model(net, filename):
    """Save trained model."""
    if not os.path.exists(params.model_root):
        os.makedirs(params.model_root)
    torch.save(net.state_dict(),
               os.path.join(params.model_root, filename))
    print("save pretrained model to: {}".format(os.path.join(params.model_root,
                                                             filename)))
