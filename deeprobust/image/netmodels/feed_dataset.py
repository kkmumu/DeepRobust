"""
This function help to feed in train and test datasets.
Select model archtecture and seed then output corresponding model.

"""
from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F #233
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
    

def feed_dataset(data, data_dict, seedin = 100, random_train = False):
    
    torch.manual_seed(seedin)
    
    if random_train == True:
        if(data == 'MNIST'):
            train_set = datasets.MNIST('./', train=True, download = True,transform=transforms.Compose([transforms.ToTensor()]))
            test_set = datasets.MNIST('../data', train=False, download = True,transform=transforms.Compose([transforms.ToTensor()]))
            full_set = torch.utils.data.ConcatDataset([train_set,test_set])
            
            trans = transforms.Compose(transforms = [
                transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
            
            train_len = 60000
            test_len = 10000
            trainset_new, testset_new = torch.utils.data.random_split(full_set,[train_len, test_len])
            trainset_new.transform = trans
            testset_new.transform = trans
            train_loader = torch.utils.data.DataLoader(trainset_new, batch_size = 64, shuffle = True)
            test_loader = torch.utils.data.DataLoader(testset_new, batch_size = 1000, shuffle = True)
            
        else:
            pass
        
        return train_loader, test_loader
    
    else:
        if(data == 'CIFAR10'):
            transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=5),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])
        
            transform_val = transforms.Compose([
                    transforms.ToTensor(),
                    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])
        
            train_loader = torch.utils.data.DataLoader(
                     datasets.CIFAR10(data_dict, train=True, download = True,
                            transform=transform_train),
                     batch_size= 1000, shuffle=True) #, **kwargs)
        
            test_loader  = torch.utils.data.DataLoader(
                     datasets.CIFAR10(data_dict, train=False, download = True,
                            transform=transform_val),
                    batch_size= 1000, shuffle=True) #, **kwargs)
        
        elif(data == 'MNIST'):
            train_loader = torch.utils.data.DataLoader(
                     datasets.MNIST(data_dict, train=True, download = True,
                     transform=transforms.Compose([transforms.ToTensor(),
                     transforms.Normalize((0.1307,), (0.3081,))])),
                     batch_size=64,
                     shuffle=True)
        
            test_loader = torch.utils.data.DataLoader(
                    datasets.MNIST('../data', train=False, download = True,
                    transform=transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))])),
                    batch_size=1000,
                    shuffle=True)
        
        elif(data == 'ImageNet'):
            pass
        
        return train_loader, test_loader
