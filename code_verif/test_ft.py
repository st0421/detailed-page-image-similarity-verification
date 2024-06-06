# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math
from tqdm import tqdm
from model import ft_net
from utils import fuse_all_conv_bn
version =  torch.__version__


######################################################################
if __name__ == '__main__':

# Options
# --------

    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--which_epoch',default='030', type=str, help='0,1,2,3...or last')
    parser.add_argument('--test_dir',default='../represent',type=str, help='./test_data')
    parser.add_argument('--name', default='ft_ResNet50_represent_none', type=str, help='save model path')
    parser.add_argument('--batchsize', default=16, type=int, help='batchsize')
    parser.add_argument('--linear_num', default=512, type=int, help='feature dimension: 512 or default or 0 (linear=False)')
    parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
    parser.add_argument('--use_efficient', action='store_true', help='use efficient-b4' )
    parser.add_argument('--PCB', action='store_true', help='use PCB' )
    parser.add_argument('--ibn', action='store_true', help='use ibn.' )
    parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')

    opt = parser.parse_args()
    ###load config###
    # load the training config
    config_path = os.path.join('./model',opt.name,'opts.yaml')
    with open(config_path, 'r') as stream:
            config = yaml.load(stream, Loader=yaml.FullLoader) # for the new pyyaml via 'conda install pyyaml'
    opt.PCB = config['PCB']
    opt.use_dense = config['use_dense']
    opt.stride = config['stride']
    if 'use_swin' in config:
        opt.use_swin = config['use_swin']
    if 'use_swinv2' in config:
        opt.use_swinv2 = config['use_swinv2']
    if 'use_convnext' in config:
        opt.use_convnext = config['use_convnext']
    if 'use_efficient' in config:
        opt.use_efficient = config['use_efficient']
    if 'use_hr' in config:
        opt.use_hr = config['use_hr']

    if 'nclasses' in config: # tp compatible with old config files
        opt.nclasses = config['nclasses']
    else: 
        opt.nclasses = 751 

    if 'ibn' in config:
        opt.ibn = config['ibn']
    if 'linear_num' in config:
        opt.linear_num = config['linear_num']

    str_ids = opt.gpu_ids.split(',')
    #which_epoch = opt.which_epoch
    name = opt.name
    test_dir = opt.test_dir

    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >=0:
            gpu_ids.append(id)

    print('We use the scale: %s'%opt.ms)
    str_ms = opt.ms.split(',')
    ms = []
    for s in str_ms:
        s_f = float(s)
        ms.append(math.sqrt(s_f))

    # set gpu ids
    if len(gpu_ids)>0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True

    ######################################################################
    # Load Data
    # ---------
    #
    # We will use torchvision and torch.utils.data packages for loading the
    # data.
    #
    #h, w = 780, 390
    h, w = 1950, 390

    data_transforms = transforms.Compose([
            transforms.Resize((h, w), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ############### Ten Crop        
            #transforms.TenCrop(224),
            #transforms.Lambda(lambda crops: torch.stack(
            #   [transforms.ToTensor()(crop) 
            #      for crop in crops]
            # )),
            #transforms.Lambda(lambda crops: torch.stack(
            #   [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
            #       for crop in crops]
            # ))
    ])

    if opt.PCB:
        data_transforms = transforms.Compose([
            transforms.Resize((780,390), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ])
        h, w = 780, 390


    data_dir = test_dir
    gallery_name = 'gallery'
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in [gallery_name,'query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                shuffle=False, num_workers=2) for x in [gallery_name,'query']}
    class_names = image_datasets['query'].classes
    use_gpu = torch.cuda.is_available()

    ######################################################################
    # Load model
    #---------------------------
    def load_network(network):
        save_path = os.path.join('./model',name,'net_%s.pth'%opt.which_epoch)
        try:
            network.load_state_dict(torch.load(save_path))
        except: 
            if torch.cuda.get_device_capability()[0]>6 and len(opt.gpu_ids)==1 and int(version[0])>1: # should be >=7
                print("Compiling model...")
                # https://huggingface.co/docs/diffusers/main/en/optimization/torch2.0
                torch.set_float32_matmul_precision('high')
                network = torch.compile(network, mode="default", dynamic=True) # pytorch 2.0
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

    def extract_feature(model,dataloaders):
        #features = torch.FloatTensor()
        # count = 0
        pbar = tqdm()
        if opt.linear_num <= 0:
            if opt.use_swin or opt.use_swinv2 or opt.use_dense or opt.use_convnext:
                opt.linear_num = 1024
            elif opt.use_efficient:
                opt.linear_num = 1792
            else:
                opt.linear_num = 2048

        for iter, data in enumerate(dataloaders):
            img, label = data
            n, c, h, w = img.size()
            # count += n
            # print(count)
            pbar.update(n)
            ff = torch.FloatTensor(n,opt.linear_num).zero_().cuda()

            if opt.PCB:
                ff = torch.FloatTensor(n,2048,6).zero_().cuda() # we have six parts

            for i in range(2):
                if(i==1):
                    img = fliplr(img)
                input_img = Variable(img.cuda())
                for scale in ms:
                    if scale != 1:
                        # bicubic is only  available in pytorch>= 1.1
                        input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                    outputs = model(input_img) 
                    ff += outputs
            # norm feature
            if opt.PCB:
                # feature size (n,2048,6)
                # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
                # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6) 
                ff = ff.div(fnorm.expand_as(ff))
                ff = ff.view(ff.size(0), -1)
            else:
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                ff = ff.div(fnorm.expand_as(ff))

            
            if iter == 0:
                features = torch.FloatTensor( len(dataloaders.dataset), ff.shape[1])
            #features = torch.cat((features,ff.data.cpu()), 0)
            start = iter*opt.batchsize
            end = min( (iter+1)*opt.batchsize, len(dataloaders.dataset))
            features[ start:end, :] = ff
        pbar.close()
        return features

    def get_id(img_path):
        labels = []
        for path, v in img_path:
            #filename = path.split('/')[-1]
            filename = os.path.basename(path)
            #label = filename.split('-')[1]+'99'+filename.split('-')[-1].split('.')[0]
            label = filename.split('.')[0][6:].strip()
            labels.append(label)
        return labels

    gallery_path = image_datasets[gallery_name].imgs
    query_path = image_datasets['query'].imgs

    gallery_label = get_id(gallery_path)
    #query_label = get_id(query_path)

    ######################################################################
    # Load Collected data Trained model
    print('-------test-----------')
    model_structure = ft_net(opt.nclasses, stride = opt.stride, linear_num=opt.linear_num)

    model = load_network(model_structure)
    model.classifier.classifier = nn.Sequential()

    # Change to test mode
    model = model.eval()
    if use_gpu:
        model = model.cuda()


    print('Here I fuse conv and bn for faster inference, and it does not work for transformers. Comment out this following line if you do not want to fuse conv&bn.')
    model = fuse_all_conv_bn(model)

    # We can optionally trace the forward method with PyTorch JIT so it runs faster.
    # To do so, we can call `.trace` on the reparamtrized module with dummy inputs
    # expected by the module.
    # Comment out this following line if you do not want to trace.
    #dummy_forward_input = torch.rand(opt.batchsize, 3, h, w).cuda()
    #model = torch.jit.trace(model, dummy_forward_input)

    #print(model)
    # Extract feature
    since = time.time()
    with torch.no_grad():
        gallery_feature = extract_feature(model,dataloaders[gallery_name])
        #query_feature = extract_feature(model,dataloaders['query'])
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.2f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
    # Save to Matlab for check
    #result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'query_f':query_feature.numpy(),'query_label':query_label}
    result = {'feat':gallery_feature.numpy(),'name':gallery_label}
    #name1 = opt.name.split('_')[-2]
    #name2 = opt.name.split('_')[-1]
    #scipy.io.savemat('mat/synthesis_test_%s_%s_%s.mat'%(name1, name2, opt.which_epoch),result)
    scipy.io.savemat('mat/ttttttttttttttttttt.mat',result)

    print(opt.name)
    result = './model/%s/result.txt'%opt.name
    #os.system('python evaluate_gpu.py | tee -a %s'%result)