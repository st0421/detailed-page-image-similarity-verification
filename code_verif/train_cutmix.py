# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#from PIL import Image
import time
import os
import collections
from tqdm import tqdm
from model import ft_net, ft_net_dense, ft_net_hr, ft_net_swin, ft_net_swinv2, ft_net_convnext, ft_net_efficient, ft_net_NAS, PCB
from random_erasing import RandomErasing
from dgfolder import DGFolder
import yaml
from shutil import copyfile
from circle_loss import CircleLoss, convert_label_to_similarity
from instance_loss import InstanceLoss
from utils import save_network
version =  torch.__version__

from pytorch_metric_learning import losses, miners #pip install pytorch-metric-learning

import cv2
from torchvision.utils import save_image
from torchvision.transforms import Normalize

def rand_bbox(size, lam): # size : [B, C, W, H]
    W = size[2] # 이미지의 width
    H = size[3] # 이미지의 height
    cut_rat = np.sqrt(1. - lam)  # 패치 크기의 비율 정하기
    cut_w = np.int32(W * cut_rat)  # 패치의 너비
    cut_h = np.int32(H * cut_rat)  # 패치의 높이

    # uniform
    # 기존 이미지의 크기에서 랜덤하게 값을 가져옵니다.(중간 좌표 추출)
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # 패치 부분에 대한 좌표값을 추출합니다.
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

######################################################################
# parser: name, dat_dir, batch,size 확인 

# h, w 수정 128번라인
######################################################################
if __name__ == '__main__':

    # Options
    # --------
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--name',default='ft_ResNet50_represent_cutmix', type=str, help='output model name')
    # data '../represent'
    parser.add_argument('--data_dir',default='../represent',type=str, help='training dir path')
    parser.add_argument('--train_all', action='store_true', help='use all training data' )
    parser.add_argument('--batchsize', default=8, type=int, help='batchsize')
    parser.add_argument('--height', default=0, type=int)    #1950 (detail) / 780 (represent) 
    parser.add_argument('--width', default=390, type=int)

    parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
    parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
    # optimizer
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay. More Regularization Smaller Weight.')
    parser.add_argument('--total_epoch', default=200, type=int, help='total training epoch')
    parser.add_argument('--fp16', action='store_true', help='use float16 instead of float32, which will save about 50%% memory' )
    parser.add_argument('--cosine', action='store_true', help='use cosine lrRate' )
    # backbone
    parser.add_argument('--linear_num', default=512, type=int, help='feature dimension: 512 or default or 0 (linear=False)')
    parser.add_argument('--stride', default=2, type=int, help='stride')
    parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
    parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
    parser.add_argument('--use_efficient', action='store_true', help='use efficientnet-b4' )
    parser.add_argument('--use_convnext', action='store_true', help='use ConvNext' )
    parser.add_argument('--ibn', action='store_true', help='use resnet+ibn' )
    parser.add_argument('--PCB', action='store_true', help='use PCB+ResNet50' )
    # loss
    parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
    parser.add_argument('--arcface', action='store_true', help='use ArcFace loss' )
    parser.add_argument('--circle', action='store_true', help='use Circle loss' )
    parser.add_argument('--cosface', action='store_true', help='use CosFace loss' )
    parser.add_argument('--contrast', action='store_true', help='use contrast loss' )
    parser.add_argument('--instance', action='store_true', help='use instance loss' )
    parser.add_argument('--ins_gamma', default=32, type=int, help='gamma for instance loss')
    parser.add_argument('--triplet', action='store_true', help='use triplet loss' )
    parser.add_argument('--lifted', action='store_true', help='use lifted loss' )
    parser.add_argument('--sphere', action='store_true', help='use sphere loss' )
    parser.add_argument('--adv', default=0.0, type=float, help='use adv loss as 1.0' )
    parser.add_argument('--aiter', default=10, type=float, help='use adv loss with iter' )

    opt = parser.parse_args()

    fp16 = opt.fp16
    data_dir = opt.data_dir

    name = opt.name
    str_ids = opt.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >=0:
            gpu_ids.append(gid)
    opt.gpu_ids = gpu_ids
    # set gpu ids
    if len(gpu_ids)>0:
        #torch.cuda.set_device(gpu_ids[0])
        cudnn.enabled = True
        cudnn.benchmark = True
    ######################################################################
    # Load Data
    # ---------
    #
        
    #h, w = 1950, 390 #detail
    # h, w = 780, 390 #represent
    h = opt.height
    w = opt.width
    transform_train_list = [
            transforms.Resize((h, w), interpolation=3),
            transforms.Pad(10),
            transforms.RandomCrop((h, w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]

    transform_val_list = [
            transforms.Resize(size=(h, w),interpolation=3), #Image.BICUBIC
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]

    if opt.erasing_p>0:
        transform_train_list = transform_train_list +  [RandomErasing(probability = opt.erasing_p, mean=[0.0, 0.0, 0.0])]

    if opt.color_jitter:
        transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list

    print(transform_train_list)
    data_transforms = {
        'train': transforms.Compose( transform_train_list ),
        'val': transforms.Compose(transform_val_list),
    }


    image_datasets = {}
    image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                            data_transforms['train'])
    image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                            data_transforms['val'])

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                shuffle=True, num_workers=1, pin_memory=True,
                                                prefetch_factor=2, persistent_workers=True) # 8 workers may work faster
                for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    use_gpu = torch.cuda.is_available()

    since = time.time()
    inputs, classes = next(iter(dataloaders['train']))
    print(time.time()-since)
    ######################################################################
    # Training the model
    # ------------------

    y_loss = {} # loss history
    y_loss['train'] = []
    y_loss['val'] = []
    y_err = {}
    y_err['train'] = []
    y_err['val'] = []

    def fliplr(img):
        '''flip horizontal'''
        inv_idx = torch.arange(img.size(3)-1,-1,-1).long().cuda()  # N x C x H x W
        img_flip = img.index_select(3,inv_idx)
        return img_flip

    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        #best_model_wts = model.state_dict()
        #best_acc = 0.0
        warm_up = 0.1 # We start from the 0.1*lrRate
        warm_iteration = round(dataset_sizes['train']/opt.batchsize)*opt.warm_epoch # first 5 epoch
        
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            # print('-' * 10)
            
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train(True)  # Set model to training mode
                else:
                    model.train(False)  # Set model to evaluate mode

                # Phases 'train' and 'val' are visualized in two separate progress bars
                pbar = tqdm()
                pbar.reset(total=len(dataloaders[phase].dataset))
                ordered_dict = collections.OrderedDict(phase="", Loss="", Acc="")

                running_loss = 0.0
                running_corrects = 0.0
                # Iterate over data.
                for iter, data in enumerate(dataloaders[phase]):
                    # get the inputs
                    inputs, labels = data
        
                    now_batch_size,c,h,w = inputs.shape
                    pbar.update(now_batch_size)  # update the pbar even in the last batch
                    if now_batch_size<opt.batchsize: # skip the last batch
                        continue

                    if use_gpu:
                        inputs = inputs.cuda().detach()
                        labels = labels.cuda().detach()

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    if phase == 'val':
                        with torch.no_grad():
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                    else:
                        if np.random.random()>=0.5: # cutmix 작동될 확률      
                            lam = np.random.beta(1.0, 1.0)
                            rand_index = torch.randperm(inputs.size()[0]).to('cuda')
                            target_a = labels
                            target_b = labels[rand_index]            
                            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                            inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]

                            # for bn in range(len(inputs[0])):
                            #     save_image(inputs[bn], "test/%d_%d_%s_%s.jpg"%(iter, bn, target_a[bn].item(), target_b[bn].item()),normalize=True)

                            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
                            outputs = model(inputs)
                            loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)

                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)

                    sm = nn.Softmax(dim=1)
                    log_sm = nn.LogSoftmax(dim=1)

                    _, preds = torch.max(outputs.data, 1)

                    del inputs

                    if epoch<opt.warm_epoch and phase == 'train': 
                        warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                        loss = loss*warm_up
                        print(loss, warm_up)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    # statistics
                    if int(version[0])>0 or int(version[2]) > 3: # for the new version like 0.4.0, 0.5.0 and 1.0.0
                        running_loss += loss.item() * now_batch_size
                        ordered_dict["Loss"] = f"{loss.item():.4f}"
                    else :  # for the old version like 0.3.0 and 0.3.1
                        running_loss += loss.data[0] * now_batch_size
                        ordered_dict["Loss"] = f"{loss.data[0]:.4f}"
                    del loss
                    running_corrects += float(torch.sum(preds == labels.data))
                    # Refresh the progress bar in every batch
                    ordered_dict["phase"] = phase
                    ordered_dict[
                        "Acc"
                    ] = f"{(float(torch.sum(preds == labels.data)) / now_batch_size):.4f}"
                    pbar.set_postfix(ordered_dict=ordered_dict)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects / dataset_sizes[phase]
                
                # print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                #     phase, epoch_loss, epoch_acc))
                ordered_dict["phase"] = phase
                ordered_dict["Loss"] = f"{epoch_loss:.4f}"
                ordered_dict["Acc"] = f"{epoch_acc:.4f}"
                pbar.set_postfix(ordered_dict=ordered_dict)
                pbar.close()
                
                y_loss[phase].append(epoch_loss)
                y_err[phase].append(1.0-epoch_acc)            
                # deep copy the model
                if phase == 'val' and epoch%10 == 9:
                    last_model_wts = model.state_dict()
                    if len(opt.gpu_ids)>1:
                        save_network(model.module, opt.name, epoch+1)
                    else:
                        save_network(model, opt.name, epoch+1)
                if phase == 'val':
                    draw_curve(epoch)
                if phase == 'train':
                    scheduler.step()
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        #print('Best val Acc: {:4f}'.format(best_acc)

        # load best model weights
        model.load_state_dict(last_model_wts)
        if len(opt.gpu_ids)>1:
            save_network(model.module, opt.name, 'last')
        else:
            save_network(model, opt.name, 'last')

        return model


    ######################################################################
    # Draw Curve
    #---------------------------
    x_epoch = []
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax1 = fig.add_subplot(122, title="top1err")
    def draw_curve(current_epoch):
        x_epoch.append(current_epoch)
        ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
        ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
        ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
        ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
        if current_epoch == 0:
            ax0.legend()
            ax1.legend()
        fig.savefig( os.path.join('./model',name,'train.jpg'))


    ######################################################################
    # Finetuning the convnet
    # ----------------------
    #
    # Load a pretrainied model and reset final fully connected layer.
    #

    return_feature = opt.arcface or opt.cosface or opt.circle or opt.triplet or opt.contrast or opt.instance or opt.lifted or opt.sphere

    if opt.use_dense:
        model = ft_net_dense(len(class_names), opt.droprate, opt.stride, circle = return_feature, linear_num=opt.linear_num)
    elif opt.use_efficient:
        model = ft_net_efficient(len(class_names), opt.droprate, circle = return_feature, linear_num=opt.linear_num)
    elif opt.use_convnext:
        model = ft_net_convnext(len(class_names), opt.droprate, circle = return_feature, linear_num=opt.linear_num)
    else:
        model = ft_net(len(class_names), opt.droprate, opt.stride, circle = return_feature, ibn=opt.ibn, linear_num=opt.linear_num)

    if opt.PCB:
        model = PCB(len(class_names))

    opt.nclasses = len(class_names)
    #print(model)
    # model to gpu
    model = model.cuda()

    optim_name = optim.SGD #apex.optimizers.FusedSGD

    if torch.cuda.get_device_capability()[0]>6 and len(opt.gpu_ids)==1 and int(version[0])>1: # should be >=7 and one gpu
        torch.set_float32_matmul_precision('high')
        print("Compiling model... The first epoch may be slow, which is expected!")
        # https://huggingface.co/docs/diffusers/main/en/optimization/torch2.0
        model = torch.compile(model, mode="reduce-overhead", dynamic = True) # pytorch 2.0

    if len(opt.gpu_ids)>1:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids) 
        if not opt.PCB:
            ignored_params = list(map(id, model.module.classifier.parameters() ))
            base_params = filter(lambda p: id(p) not in ignored_params, model.module.parameters())
            classifier_params = model.module.classifier.parameters()
            optimizer_ft = optim_name([
                {'params': base_params, 'lr': 0.1*opt.lr},
                {'params': classifier_params, 'lr': opt.lr}
            ], weight_decay=opt.weight_decay, momentum=0.9, nesterov=True)
        else:
            ignored_params = list(map(id, model.module.model.fc.parameters() ))
            ignored_params += (list(map(id, model.module.classifier0.parameters() ))
                        +list(map(id, model.module.classifier1.parameters() ))
                        +list(map(id, model.module.classifier2.parameters() ))
                        +list(map(id, model.module.classifier3.parameters() ))
                        +list(map(id, model.module.classifier4.parameters() ))
                        +list(map(id, model.module.classifier5.parameters() ))
                        #+list(map(id, model.module.classifier6.parameters() ))
                        #+list(map(id, model.module.classifier7.parameters() ))
                        )
            base_params = filter(lambda p: id(p) not in ignored_params, model.module.parameters())
            classifier_params = filter(lambda p: id(p) in ignored_params, model.module.parameters())
            optimizer_ft = optim_name([
                {'params': base_params, 'lr': 0.1*opt.lr},
                {'params': classifier_params, 'lr': opt.lr}
            ], weight_decay=opt.weight_decay, momentum=0.9, nesterov=True)
    else:
        if not opt.PCB:
            ignored_params = list(map(id, model.classifier.parameters() ))
            base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
            classifier_params = model.classifier.parameters()
            optimizer_ft = optim_name([
                {'params': base_params, 'lr': 0.1*opt.lr},
                {'params': classifier_params, 'lr': opt.lr}
            ], weight_decay=opt.weight_decay, momentum=0.9, nesterov=True)
        else:
            ignored_params = list(map(id, model.model.fc.parameters() ))
            ignored_params += (list(map(id, model.classifier0.parameters() )) 
                        +list(map(id, model.classifier1.parameters() ))
                        +list(map(id, model.classifier2.parameters() ))
                        +list(map(id, model.classifier3.parameters() ))
                        +list(map(id, model.classifier4.parameters() ))
                        +list(map(id, model.classifier5.parameters() ))
                        #+list(map(id, model.classifier6.parameters() ))
                        #+list(map(id, model.classifier7.parameters() ))
                        )
            base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
            classifier_params = filter(lambda p: id(p) in ignored_params, model.parameters())
            optimizer_ft = optim_name([
                {'params': base_params, 'lr': 0.1*opt.lr},
                {'params': classifier_params, 'lr': opt.lr}
            ], weight_decay=opt.weight_decay, momentum=0.9, nesterov=True)

    # Decay LR by a factor of 0.1 every 40 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=opt.total_epoch*2//3, gamma=0.1)
    if opt.cosine:
        exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, opt.total_epoch, eta_min=0.01*opt.lr)

    ######################################################################
    # Train and evaluate
    # ^^^^^^^^^^^^^^^^^^
    #
    # It should take around 1-2 hours on GPU. 
    #
    dir_name = os.path.join('./model',name)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    #record every run
    copyfile('./train.py', dir_name+'/train.py')
    copyfile('./model.py', dir_name+'/model.py')

    # save opts
    with open('%s/opts.yaml'%dir_name,'w') as fp:
        yaml.dump(vars(opt), fp, default_flow_style=False)

    criterion = nn.CrossEntropyLoss()

    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                        num_epochs=opt.total_epoch)

