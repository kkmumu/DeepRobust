import requests
import torch
from torchvision import datasets,models,transforms
import torch.nn.functional as F
from sklearn.model_selection import KFold
import os

import numpy as np
import argparse
import matplotlib.pyplot as plt
import random

import deeprobust.image.utils
from deeprobust.image.attack.fgsm import FGSM

def run_attack(attackmethod, batch_size, batch_num, device, test_loader, random_targeted = False, target_label = -1, **kwargs):
    test_loss = 0
    correct = 0
    samplenum = 1000
    count = 0
    classnum = 10
    max_error = 0
    
    for count, (data, target) in enumerate(test_loader):
        if count == batch_num:
            break
        print('batch:{}'.format(count))

        data, target = data.to(device), target.to(device)
        if(random_targeted == True):
            r = list(range(0, target)) + list(range(target+1, classnum))
            target_label = random.choice(r)
            adv_example = attackmethod.generate(data, target, target_label = target_label, **kwargs)

        elif(target_label >= 0):
            adv_example = attackmethod.generate(data, target, target_label = target_label, **kwargs)

        else:
            adv_example = attackmethod.generate(data, target, **kwargs)

        output = model(adv_example)
        test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        
        # update the maximal loss error between test samples and adversarial attack samples
        if abs(F.nll_loss(output, target).item()-F.nll_loss(data, target).item()) > max_error:
            max_error = abs(F.nll_loss(output, target).item()-F.nll_loss(data, target).item())

        pred = output.argmax(dim = 1, keepdim = True)  # get the index of the max log-probability.

        correct += pred.eq(target.view_as(pred)).sum().item()

    batch_num = count+1
    test_loss /= len(test_loader.dataset)
    
    print("===== ACCURACY =====")
    print('Attack Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, batch_num * batch_size,
        100. * correct / (batch_num * batch_size)))
    
    return max_error

def generate_perturbation(attackmethod, batch_size, batch_num, device, train_loader, epsilon, random_targeted = False, target_label = -1):
    test_loss = 0
    correct = 0
    samplenum = 1000
    count = 0
    classnum = 10
    max_error = 0
    
    for count, (data, target) in enumerate(train_loader):
        if count == batch_num:
            break
        print('batch:{}'.format(count))

        data, target = data.to(device), target.to(device)
        if(random_targeted == True):
            r = list(range(0, target)) + list(range(target+1, classnum))
            target_label = random.choice(r)
            adv_example = attackmethod.generate(data, target, target_label = target_label, epsilon = epsilon)

        elif(target_label >= 0):
            adv_example = attackmethod.generate(data, target, target_label = target_label, epsilon = epsilon)

        else:
            adv_example = attackmethod.generate(data, target, epsilon = epsilon)

        output = model(adv_example)
        test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        
        # update the maximal loss error between test samples and adversarial attack samples
        if abs(F.nll_loss(output, target).item()-F.nll_loss(data, target).item()) > max_error:
            max_error = abs(F.nll_loss(output, target).item()-F.nll_loss(data, target).item())

        pred = output.argmax(dim = 1, keepdim = True)  # get the index of the max log-probability.

        correct += pred.eq(target.view_as(pred)).sum().item()

    batch_num = count+1
    test_loss /= len(train_loader.dataset)
    
    print("===== ACCURACY =====")
    print('Perturbation Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, batch_num * batch_size,
        100. * correct / (batch_num * batch_size)))
    
    return max_error


    
def evaluate_perturbation(n_splits, attackmethod, batch_size, batch_num, device, train_loader, epsilon, random_targeted = False, target_label = -1):
    
    kfold = KFold(n_splits)
    max_perturb = 0
    
    ### Dividing data and targets into folds
    for  fold, (train_index, test_index) in enumerate(kfold.split(train_loader.dataset)):
        
        data_test_fold = train_loader.dataset.data[test_index]
        targets_test_fold = train_loader.dataset.targets[test_index]
        
        test = torch.utils.data.TensorDataset(data_test_fold, targets_test_fold)
        test = test.unsqueeze(0)
        test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)
        
        max_error = generate_perturbation(attackmethod, batch_size, batch_num, device, test_loader, epsilon, random_targeted, target_label)
        print(max_error)
        
        if max_error > max_perturb:
            max_perturb = max_error
            
    print("PERTURBATION ERROR: %.4f\n",max_perturb)
    return max_perturb

        
def load_net(attack_model, filename, path):
    if(attack_model == "CNN"):
        from deeprobust.image.netmodels.CNN import Net

        model = Net()
    if(attack_model == "ResNet18"):
        import deeprobust.image.netmodels.resnet as Net
        model = Net.ResNet18()

    model.load_state_dict(torch.load(path + filename))
    model.eval()
    return model

def generate_dataloader(dataset, batch_size):
    if(dataset == "MNIST"):
        test_loader = torch.utils.data.DataLoader(
                        datasets.MNIST('deeprobust/image/data', train = False,
                        download = True,
                        transform = transforms.Compose([transforms.ToTensor()])),
                        batch_size = args.batch_size,
                        shuffle = True)
        print("Loading MNIST dataset.")

    elif(dataset == "CIFAR" or args.dataset == 'CIFAR10'):
        test_loader = torch.utils.data.DataLoader(
                        datasets.CIFAR10('deeprobust/image/data', train = False,
                        download = True,
                        transform = transforms.Compose([transforms.ToTensor()])),
                        batch_size = args.batch_size,
                        shuffle = True)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        print("Loading CIFAR10 dataset.")

    elif(dataset == "ImageNet"):
        test_loader = torch.utils.data.DataLoader(
                        datasets.CIFAR10('deeprobust/image/data', train=False,
                        download = True,
                        transform = transforms.Compose([transforms.ToTensor()])),
                        batch_size = args.batch_size,
                        shuffle = True)
        print("Loading ImageNet dataset.")
    return test_loader

def parameter_parser():
    parser = argparse.ArgumentParser(description = "Run attack algorithms.", usage ='Use -h for more information.')

    parser.add_argument("--attack_method",
                        default = 'FGSM',
                        help = "Choose a attack algorithm from: PGD(default), FGSM, LBFGS, CW, deepfool, onepixel, Nattack")
    parser.add_argument("--attack_model",
                        default = "CNN",
                        help = "Choose network structure from: CNN, ResNet")
    parser.add_argument("--path",
                        default = "./trained_models/",
                        help = "Type the path where the model is saved.")
    parser.add_argument("--file_name",
                        default = 'MNIST_CNN_epoch_20.pt',
                        help = "Type the file_name of the model that is to be attack. The model structure should be matched with the ATTACK_MODEL parameter.")
    parser.add_argument("--dataset",
                        default = 'MNIST',
                        help = "Choose a dataset from: MNIST(default), CIFAR(or CIFAR10), ImageNet")
    parser.add_argument("--epsilon", type = float, default = 0.3)
    parser.add_argument("--batch_num", type = int, default = 1000)
    parser.add_argument("--batch_size", type = int, default = 1000)
    parser.add_argument("--num_steps", type = int, default = 40)
    parser.add_argument("--step_size", type = float, default = 0.01)
    parser.add_argument("--random_targeted", type = bool, default = False,
                        help = "default: False. By setting this parameter be True, the program would random generate target labels for the input samples.")
    parser.add_argument("--target_label", type = int, default = -1,
                        help = "default: -1. Generate all attack Fixed target label.")
    parser.add_argument("--device", default = 'cuda',
                        help = "Choose the device.")

    return parser.parse_args()


if __name__ == "__main__":
    # load train set
    train_loader = torch.utils.data.DataLoader(
             datasets.MNIST('./', train=True, download = False,
             transform=transforms.Compose([transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))])), # 0.1307 and 0.3081 is the mean and std of MNIST
             batch_size=64,
             shuffle=True)
    
    # download example model
    example_model_path = './trained_models/MNIST_CNN_epoch_20.pt'
    if not (os.path.exists('./trained_models')):
        os.mkdir('./trained_models')
        print('create path: ./trained_models')
    model_url = "https://github.com/I-am-Bot/deeprobust_trained_model/blob/master/MNIST_CNN_epoch_20.pt?raw=true"
    r = requests.get(model_url)
    print('Downloading example model...')
    with open(example_model_path,'wb') as f:
        f.write(r.content)
    print('Downloaded.') 
    
       
    # set parameters
    n_splits = 10
    batch_size = 64
    batch_num = 1000
    device = 'cuda'
    epsilon = 0.3
    attack_model = 'CNN'
    file_name = 'MNIST_CNN_epoch_20.pt'
    path = './trained_models/'
    model = load_net(attack_model, file_name, path)
    attack_method = FGSM(model, device)
    
    evaluate_perturbation(n_splits, attack_method, batch_size, batch_num, device, train_loader, epsilon=epsilon)


    
    # # read arguments
    # args = parameter_parser() # read argument and creat an argparse object

    # # download example model
    # example_model_path = './trained_models/MNIST_CNN_epoch_20.pt'
    # if not (os.path.exists('./trained_models')):
    #     os.mkdir('./trained_models')
    #     print('create path: ./trained_models')
    # model_url = "https://github.com/I-am-Bot/deeprobust_trained_model/blob/master/MNIST_CNN_epoch_20.pt?raw=true"
    # r = requests.get(model_url)
    # print('Downloading example model...')
    # with open(example_model_path,'wb') as f:
    #     f.write(r.content)
    # print('Downloaded.')
    # # load model
    # model = load_net(args.attack_model, args.file_name, args.path)

    # print("===== START ATTACK =====")
    # if(args.attack_method == "PGD"):
    #     from deeprobust.image.attack.pgd import PGD
    #     test_loader = generate_dataloader(args.dataset, args.batch_size)
    #     attack_method = PGD(model, args.device)
    #     utils.tab_printer(args)
    #     run_attack(attack_method, args.batch_size, args.batch_num, args.device, test_loader, epsilon = args.epsilon)

    # elif(args.attack_method == "FGSM"):
    #     from deeprobust.image.attack.fgsm import FGSM
    #     test_loader = generate_dataloader(args.dataset, args.batch_size)
    #     attack_method = FGSM(model, args.device)
    #     utils.tab_printer(args)
    #     #run_attack(attack_method, args.batch_size, args.batch_num, args.device, test_loader, epsilon = args.epsilon)
    #     evaluate_perturbation(10, attack_method, args.batch_size, args.batch_num, args.device, test_loader, epsilon = args.epsilon)

    # elif(args.attack_method == "LBFGS"):
    #     from deeprobust.image.attack.lbfgs import LBFGS
    #     try:
    #         if (args.batch_size >1):
    #             raise ValueError("batch_size shouldn't be larger than 1.")
    #     except ValueError:
    #         args.batch_size = 1

    #     try:
    #         if (args.random_targeted == 0 and args.target_label == -1):
    #             raise ValueError("No target label assigned. Random generate target for each input.")
    #     except ValueError:
    #         args.random_targeted = True

    #     utils.tab_printer(args)
    #     test_loader = generate_dataloader(args.dataset, args.batch_size)
    #     attack_method = LBFGS(model, args.device)
    #     run_attack(attack_method, 1, args.batch_num, args.device, test_loader, random_targeted = args.random_targeted, target_label = args.target_label)

    # elif(args.attack_method == "CW"):
    #     from deeprobust.image.attack.cw import CarliniWagner
    #     attack_method = CarliniWagner(model, args.device)
    #     try:
    #         if (args.batch_size > 1):
    #             raise ValueError("batch_size shouldn't be larger than 1.")
    #     except ValueError:
    #         args.batch_size = 1

    #     try:
    #         if (args.random_targeted == 0 and args.target_label == -1):
    #             raise ValueError("No target label assigned. Random generate target for each input.")
    #     except ValueError:
    #         args.random_targeted = True

    #     utils.tab_printer(args)
    #     test_loader = generate_dataloader(args.dataset, args.batch_size)
    #     run_attack(attack_method, 1, args.batch_num, args.device, test_loader, random_targeted = args.random_targeted, target_label = args.target_label)

    # elif(args.attack_method == "deepfool"):
    #     from deeprobust.image.attack.deepfool import DeepFool
    #     attack_method = DeepFool(model, args.device)
    #     try:
    #         if (args.batch_size > 1):
    #             raise ValueError("batch_size shouldn't be larger than 1.")
    #     except ValueError:
    #         args.batch_size = 1

    #     utils.tab_printer(args)
    #     test_loader = generate_dataloader(args.dataset, args.batch_size)
    #     run_attack(attack_method, args.batch_size, args.batch_num, args.device, test_loader)

    # elif(args.attack_method == "onepixel"):
    #     from deeprobust.image.attack.onepixel import Onepixel
    #     attack_method = Onepixel(model, args.device)
    #     try:
    #         if (args.batch_size > 1):
    #             raise ValueError("batch_size shouldn't be larger than 1.")
    #     except ValueError:
    #         args.batch_size = 1

    #     utils.tab_printer(args)
    #     test_loader = generate_dataloader(args.dataset, args.batch_size)
    #     run_attack(attack_method, args.batch_size, args.batch_num, args.device, test_loader)

    # elif(args.attack_method == "Nattack"):
    #     pass
