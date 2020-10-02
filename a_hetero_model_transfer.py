from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from PIL import Image
import time
import os
import random
from torch.nn import functional as F
from torch.nn import init
from torchvision import models
import random
# import pdb


def transfer_from_hetero_model(model1, model2, dis='eu'):
    if isinstance(model1, nn.DataParallel):
        model1 = model1.module

    if isinstance(model2, nn.DataParallel):
        model2 = model2.module


    model1_paras, model2_paras=[],[]
    name1,name2=[],[]
    # for module in model1.model.modules():
    for name,module in model1.named_modules():
        # pdb.set_trace()
        # print('module:',module)
        if isinstance(module, nn.Conv2d):
            # name1.append(name)
            for p in module.parameters():
                model1_paras.append(p)
                name1.append(name)

    for name,module in model2.named_modules():
        if isinstance(module, nn.Conv2d):
            # name2.append(name)
            for p in module.parameters():
                model2_paras.append(p) 
                name2.append(name)


    transferable_layer_num = min((len(model1_paras), len(model2_paras)))
    # pdb.set_trace()

    # transfer
    #32, 3, 3, 3
    #32, 3, 5, 5
    m = nn.ZeroPad2d(1)
    for i in range(transferable_layer_num):
        p1 = model1_paras[i]
        p2 = model2_paras[i]
        # print('name1:',name1[i])
        # print('name2:',name2[i])
        # print('p1 size:',p1.size())
        # print('p2 size:',p2.size())

        if len(p1.data.size())==1:
            p1.data = p2.data
        else:
            p1.data = m(p2.data)
        # print('after p1 size:',p1.size())
        # pdb.set_trace()

        # p1.size():16,3,3,64
        # p2.size():8,5,5,32

    return  model1



def circle_copy(arr1, arr2, arr_dims):
    c1 = arr1.size(0)
    c2 = arr2.size(0)
    # i=0
    for i in range(c1):# i<c1:
        j=i%c2
        if arr_dims==1:
            arr1[i] = arr2[j]
        else:
            arr1[i,:,:] = arr2[j,:,:]
        # i=i+1
    return arr1

def copy_and_padding(arr1, arr2, arr_dims):
    c1 = arr1.size(0)
    c2 = arr2.size(0)
    # i=0
    for i in range(c1):# i<c1:
        # j=i%c2
        if arr_dims==1:
            if i>=c1:
                arr1[i]= torch.zeros(1,dtype=torch.FloatTensor).cuda()
            else:
                arr1[i] = arr2[i]
        else:
            c,h,w=arr2.size()
            if i>=c1:
                arr1[i,:,:]= torch.zeros((h,w),dtype=torch.FloatTensor).cuda()
            else:
                arr1[i,:,:] = arr2[i,:,:]          
        # i=i+1
    return arr1

# model 1 is a target model, model 2 is a pre-trained model
def transfer_from_hetero_model_more(model1, model2, dis='eu'):
    if isinstance(model1, nn.DataParallel):
        model1 = model1.module
    if isinstance(model2, nn.DataParallel):
        model2 = model2.module

    model1_paras, model2_paras=[],[]
    model1_bias, model2_bias=[],[]
    name1,name2=[],[]
    # for module in model1.model.modules():
    for name,module in model1.named_modules():
        # pdb.set_trace()
        # print('module:',module)
        # if (isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d)) and 'downsample' not in name.split('.'):
        if isinstance(module, nn.Conv2d)  and 'downsample' not in name.split('.'):
            # name1.append(name)
            # for p in module.parameters():
            for (n, p) in module.named_parameters():
                if 'bias' in n.split('.'):
                    continue
                model1_paras.append(p)
                name1.append(name+'_'+n)
    
    for name,module in model2.named_modules():
        # if (isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d)) and 'downsample' not in name.split('.'):
        if isinstance(module, nn.Conv2d)  and 'downsample' not in name.split('.'):
            # name2.append(name)
            # for p in module.parameters():
            for (n, p) in module.named_parameters():
                if 'bias' in n.split('.'):
                    continue                
                model2_paras.append(p) 
                name2.append(name+'_'+n)
    print('name1:',name1)
    print('name2:',name2)
    transferable_layer_num = min((len(model1_paras), len(model2_paras)))
    # transfer
    #32, 3, 3, 3
    #32, 3, 5, 5
    
    for i in range(transferable_layer_num):
        p1 = model1_paras[i]
        p2 = model2_paras[i]
        print('name1:',name1[i])
        print('name2:',name2[i])
        # print('p1 size:',p1.size())
        # print('p2 size:',p2.size())


        if len(p1.data.size())==1 and len(p2.data.size())==1:
            c_out1= p1.size()[0]
            c_out2 = p2.size()[0]
            p1.data=circle_copy(p1.data,p2.data,1)
        else:
            c_out1,c_in1, k_h1, c_w1 = p1.size()
            c_out2,c_in2, k_h2, c_w2 = p2.size()
            assert(k_h1== c_w1)
            assert(k_h2== c_w2)
            # step 1: transform the 2D kernal size
            if k_h1>=k_h2:
                # m = nn.ZeroPad2d(int((k_h1-k_h2)/2))
                # p2_resized = m(p2.data)                   ### 3x3--->5x5, zero padding
                # p2_resized=p2.data.resize_(k_h2,c_w2)       ### 3x3--->5x5, resize
                p2_resized = F.interpolate(p2.data, size=[k_h1, c_w1], mode="bilinear", align_corners=True)  #nearest,bilinear,align_corners=True
            else:
                # p1.data = p2.data
                p2_resized = F.interpolate(p2.data, size=[k_h1, c_w1], mode="bilinear", align_corners=True)
                # p2_resized=p2.data.resize_(k_h2,c_w2)      
            # step 2: 
            for j in range(c_out1):
                k=j%c_out2
                p1.data[j,:,:,:]=circle_copy(p1.data[j,:,:,:],p2_resized[k,:,:,:],3)   
        
        # norm? 20200831
        # p1.data = p1.data*torch.numel(p2.data)/torch.numel(p1.data)
        # print('after p1 size:',p1.size())
    return  model1

# verify the property of random permutation  
# model 1 is a target model, model 2 is a pre-trained model
def transfer_from_hetero_model_more_interval(model1, model2, dis='eu'):
    if isinstance(model1, nn.DataParallel):
        model1 = model1.module
    if isinstance(model2, nn.DataParallel):
        model2 = model2.module

    model1_paras, model2_paras=[],[]
    model1_bias, model2_bias=[],[]
    name1,name2=[],[]
    # for module in model1.model.modules():
    for name,module in model1.named_modules():
        # pdb.set_trace()
        # print('module:',module)
        # if (isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d)) and 'downsample' not in name.split('.'):
        if isinstance(module, nn.Conv2d)  and 'downsample' not in name.split('.'):
            # name1.append(name)
            # for p in module.parameters():
            for (n, p) in module.named_parameters():
                if 'bias' in n.split('.'):
                    continue
                model1_paras.append(p)
                name1.append(name+'_'+n)
    
    for name,module in model2.named_modules():
        # if (isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d)) and 'downsample' not in name.split('.'):
        if isinstance(module, nn.Conv2d)  and 'downsample' not in name.split('.'):
            # name2.append(name)
            # for p in module.parameters():
            for (n, p) in module.named_parameters():
                if 'bias' in n.split('.'):
                    continue                
                model2_paras.append(p) 
                name2.append(name+'_'+n)
    
    print('name1:',name1)
    print('name2:',name2)
    # indexes = range(len(model2_paras))
    # random.shuffle(indexes)
    indexes = [3,1,0,7,4,6,2]          # for random permutation
    # indexes = [0,2,4,6]            # for continuous verification
    print(indexes)
    model2_paras=[model2_paras[i] for i in indexes]

    transferable_layer_num = min((len(model1_paras), len(model2_paras)))
    # transfer
    #32, 3, 3, 3
    #32, 3, 5, 5
    
    for i in range(transferable_layer_num):
        p1 = model1_paras[i]
        p2 = model2_paras[i]
        print('name1:',name1[i])
        print('name2:',name2[i])
        # print('p1 size:',p1.size())
        # print('p2 size:',p2.size())


        if len(p1.data.size())==1 and len(p2.data.size())==1:
            c_out1= p1.size()[0]
            c_out2 = p2.size()[0]
            p1.data=circle_copy(p1.data,p2.data,1)
        else:
            c_out1,c_in1, k_h1, c_w1 = p1.size()
            c_out2,c_in2, k_h2, c_w2 = p2.size()
            assert(k_h1== c_w1)
            assert(k_h2== c_w2)
            # step 1: transform the 2D kernal size
            if k_h1>=k_h2:
                # m = nn.ZeroPad2d(int((k_h1-k_h2)/2))
                # p2_resized = m(p2.data)                   ### 3x3--->5x5, zero padding
                # p2_resized=p2.data.resize_(k_h2,c_w2)       ### 3x3--->5x5, resize
                p2_resized = F.interpolate(p2.data, size=[k_h1, c_w1], mode="bilinear",align_corners=True)  #nearest,bilinear
            else:
                # p1.data = p2.data
                p2_resized = F.interpolate(p2.data, size=[k_h1, c_w1], mode="bilinear",align_corners=True)
                # p2_resized=p2.data.resize_(k_h2,c_w2)      
            # step 2: 
            for j in range(c_out1):
                k=j%c_out2
                p1.data[j,:,:,:]=circle_copy(p1.data[j,:,:,:],p2_resized[k,:,:,:],3)   
        # norm? 20200831
        # p1.data = p1.data*torch.numel(p2.data)/torch.numel(p1.data)
        # print('after p1 size:',p1.size())
    return  model1





