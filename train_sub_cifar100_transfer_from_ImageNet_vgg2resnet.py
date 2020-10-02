'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np

from models import *
from utils import progress_bar
# import torchvision.models as models
from torch.optim import lr_scheduler
from a_hetero_model_transfer import transfer_from_hetero_model,transfer_from_hetero_model_bn,transfer_from_hetero_model_more,transfer_from_hetero_model_padding
import pdb
import random
from compute_accuracy import accuracy

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--save_path',default='checkpoint', type=str, help='output model name')
parser.add_argument('--load_path',default='checkpoint', type=str, help='output model name')
parser.add_argument('--resume', '-r', action='store_true',help='resume from checkpoint')
parser.add_argument('--use_pretrain', action='store_true', help='use pretrained model' )
parser.add_argument('--reduce_to_baseline', action='store_true', help='use pretrained model' )
args = parser.parse_args()

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

dataset_size=50000
split=1000
indices = list(range(dataset_size))
np.random.shuffle(indices)
train_indices = indices[:split]
print('10 example indices:',train_indices[:10])

# Creating PT data samplers and loaders:
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, sampler=train_sampler)

testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('test args.use_pretrain:',args.use_pretrain)
if not args.use_pretrain:
    net = resnet8(num_classes=100)
    print('net:',net)
else:
    net0 =torchvision.models.vgg11_bn(pretrained=True)
    net = resnet8(num_classes=100)
    print('net:',net)
    # net = torch.nn.DataParallel(net)
    print('test args.reduce_to_baseline:',args.reduce_to_baseline)
    if not args.reduce_to_baseline:
        print('****************transfer**************')
        net = transfer_from_hetero_model_more(net, net0)          ### remove this line for comparison

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoints/'+args.save_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('checkpoints/'+args.save_path+'/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[150,250], gamma=0.1) #### add
# Training

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    lr_scheduler.step()              ############add 
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    output_all = torch.FloatTensor(10000,100).zero_().cuda()
    target_all = torch.LongTensor(10000,).zero_().cuda()
    # ff = torch.FloatTensor(n,1536).zero_()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            output_all[batch_idx*100:(batch_idx+1)*100,:] = outputs
            target_all[batch_idx*100:(batch_idx+1)*100] = targets

            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
        results = accuracy(output_all,target_all,(1,5,10))
        print('accuracy:',results)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoints/'+args.save_path):
            os.mkdir('checkpoints/'+args.save_path)
        torch.save(state, 'checkpoints/'+args.save_path+'/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+350):        ###########
    train(epoch)
    test(epoch)
