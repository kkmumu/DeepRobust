"""
This function help to train model of different archtecture easily. Select model archtecture and training data, then output corresponding model.

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

def train(model, data, device, maxepoch, ensemble_size, t, data_path = './', save_per_epoch = 10, seed = 199, seedin = 100, shuffle = False):
    """train.

    Parameters
    ----------
    model :
        model(option:'CNN', 'ResNet18', 'ResNet34', 'ResNet50', 'densenet', 'vgg11', 'vgg13', 'vgg16', 'vgg19')
    data :
        data(option:'MNIST','CIFAR10')
    device :
        device(option:'cpu', 'cuda')
    maxepoch :
        training epoch
    data_path :
        data path(default = './')
    save_per_epoch :
        save_per_epoch(default = 10)
    seed :
        seed
    random_test :
        set train_set to be random drawn
    Examples
    --------
    >>>import deeprobust.image.netmodels.train_model as trainmodel
    >>>trainmodel.train('CNN', 'MNIST', 'cuda', 20)
    """

    torch.manual_seed(seed)
    import deeprobust.image.netmodels.feed_dataset as fd
    train_loader, test_loader = fd.feed_dataset(data, data_path, seedin, random_train = True)
    device = torch.device("cuda")

    if (model == 'CNN'):
        import deeprobust.image.netmodels.CNN as MODEL
        #from deeprobust.image.netmodels.CNN import Net
        train_net = MODEL.Net(ensemble_size = ensemble_size, t = t).to(device = device)

    elif (model == 'ResNet18'):
        import deeprobust.image.netmodels.resnet as MODEL
        train_net = MODEL.ResNet18().to(device = device)

    elif (model == 'ResNet34'):
        import deeprobust.image.netmodels.resnet as MODEL
        train_net = MODEL.ResNet34().to(device = device)

    elif (model == 'ResNet50'):
        import deeprobust.image.netmodels.resnet as MODEL
        train_net = MODEL.ResNet50().to(device = device)

    elif (model == 'densenet'):
        import deeprobust.image.netmodels.densenet as MODEL
        train_net = MODEL.densenet_cifar().to(device = device)

    elif (model == 'vgg11'):
        import deeprobust.image.netmodels.vgg as MODEL
        train_net = MODEL.VGG('VGG11').to(device = device)
    elif (model == 'vgg13'):
        import deeprobust.image.netmodels.vgg as MODEL
        train_net = MODEL.VGG('VGG13').to(device = device)
    elif (model == 'vgg16'):
        import deeprobust.image.netmodels.vgg as MODEL
        train_net = MODEL.VGG('VGG16').to(device = device)
    elif (model == 'vgg19'):
        import deeprobust.image.netmodels.vgg as MODEL
        train_net = MODEL.VGG('VGG19').to(device = device)


    optimizer = optim.SGD(train_net.parameters(), lr=0.01, momentum=0.5)

    save_model = True
    train_loss = 0	
    test_loss = 0
    for epoch in range(1, maxepoch + 1):     ## 5 batches

        print(epoch)
        train_loss += MODEL.train(train_net, device, train_loader, optimizer, epoch)
        test_loss += MODEL.test(train_net, device, test_loader)

        if (save_model and (epoch % (save_per_epoch) == 0 or epoch == maxepoch)):
            if os.path.isdir('./trained_models/'):
                print('Save model.')
                torch.save(train_net.state_dict(), './trained_models/'+ data + "_" + model + "_epoch_" + str(epoch) + ".pt")
            else:
                os.mkdir('./trained_models/')
                print('Make directory and save model.')
                torch.save(train_net.state_dict(), './trained_models/'+ data + "_" + model + "_epoch_" + str(epoch) + ".pt")
    
    empirical_error = train_loss / (len(train_loader.dataset)*maxepoch)
    expected_error = (train_loss + test_loss) / ((len(train_loader)+len(test_loader))*maxepoch)
    generalization_error = abs(expected_error - empirical_error)

    print("========Expected Error========")
    print('Expected Error over the whole set: {:.4f}'.format(expected_error))	

    print("========Empirical Error========")
    print('Empirical Error over the training set: {:.4f}'.format(empirical_error))

    print("========Generalization Error========")
    print('Generalization Error: {:.4f}\n'.format(generalization_error))

    return generalization_error
