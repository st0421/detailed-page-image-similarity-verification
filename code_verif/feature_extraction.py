# -*- coding: utf-8 -*-

from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import transforms
import os
from PIL import Image

from code_verif.model import ft_net
from code_verif.utils import fuse_all_conv_bn
version =  torch.__version__


######################################################################
# Load model
#---------------------------
def load_network(network, name, which_epoch):
    save_path = os.path.join('D:/VILAB/AIproject/code_verif/model',name,'net_%s.pth'%which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network

######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,data_path, h, w):
        
    img = Image.open(data_path).convert('RGB')
    data_transforms = transforms.Compose([
            transforms.Resize((h, w), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = data_transforms(img)
    img = img.unsqueeze(0)
    n, c, h, w = img.size()
    ff = torch.FloatTensor(1,512).zero_().cuda()

    for i in range(2):
        if(i==1):
            img = fliplr(img)
        input_img = Variable(img.cuda())
        
        outputs = model(input_img) 
        ff += outputs
    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
    ff = ff.div(fnorm.expand_as(ff))
    feature = torch.FloatTensor(1,ff.shape[1])
    feature[:] =ff
    return feature

# Load Collected data Trained model

def make_feature(data_path, which_epoch = '030', name = 'ft_ResNet50_represent_none'):
    check = name.split('_')[2]
    if check =='represent':
        h, w = 780, 390
    else:
        h, w = 1950, 390

    # set gpu ids
    torch.cuda.set_device(0)
    cudnn.benchmark = True

    use_gpu = torch.cuda.is_available()
    model_structure = ft_net()

    model = load_network(model_structure, name, which_epoch)
    model.classifier.classifier = nn.Sequential()

    # Change to test mode
    model = model.eval()
    if use_gpu:
        model = model.cuda()

    model = fuse_all_conv_bn(model)

    with torch.no_grad():
        result = extract_feature(model,data_path, h, w)
    
    return result

