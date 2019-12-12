from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
import copy
import cv2
from dataloader import *
import argparse
from PIL import Image

def model_init(num_class=8):
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 8)
    return model_ft

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    # print(outputs, labels)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save({
                    'epoch': num_epochs,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_acc,
                }, 'checkpoint.pth')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}          val loss: {:4f}'.format(best_acc, best_loss))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model

def main_train(model, dataloaders, num_epochs=25):

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    
    model_ft = train_model(model, dataloaders, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=num_epochs)
    return model_ft

def evaluate(model, checkpoint_, dataloader, labels):
    

    checkpoint = torch.load(checkpoint_, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    accuracy = 0
    for inputs, labels in dataloaders['val']:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        accuracy += torch.sum(preds == labels.data)
    accuracy = accuracy.double()/ dataset_sizes['val']
    print('Accuracy: {:2f}'.format(accuracy))
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', help='train or evalute')
    parser.add_argument('--epochs', default=25, type=int, help='Number of epoch')
    opt = parser.parse_args()
    
    data_dir = 'dataset'
    dataloaders, dataset_sizes, class_names = data_loader(data_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    model_ft = model_init()
    model_ft = model_ft.to(device)
    if opt.mode == 'train':
        model = main_train(model_ft, dataloaders, num_epochs=opt.epochs)

    elif opt.mode == 'evalute':
            pred =  evaluate(model_ft, 'checkpoint.pth', dataloaders, class_names) 