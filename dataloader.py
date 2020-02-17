from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
# import matplotlib.pyplot as plt
import time
import os
import copy

def data_loader(data_dir='dataset'):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),  
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # data_dir = 'data/hymenoptera_data'
    data_dir = data_dir 
    print(os.listdir(data_dir))
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    class_names = {k: class_names[k] for k in range(len(class_names))}
    print(class_names)
    return dataloaders, dataset_sizes, class_names

if __name__=="__main__":
    data_dir = '/home/vietthang/ge3f/dataset'
    data_loader, dataset_sizes, class_names = data_loader(data_dir)
    print(class_names)
