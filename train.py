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
import cv2
from dataloader import *
import argparse
from PIL import Image
import numpy as np

def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=25):
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

def main_train(model, dataloaders, device, num_epochs):

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    
    model_ft = train_model(model, dataloaders, criterion, optimizer_ft, exp_lr_scheduler, device,
                       num_epochs=num_epochs)
    return model_ft

def evaluate(model, checkpoint_, dataloader, device, test_dataloader):
    

    checkpoint = torch.load(checkpoint_)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    # print(model)
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    data = '/home/vietthangtik15/cv/dataset/val/alcado'
    # img = cv2.imread('/home/vietthangtik15/cv/dataset/val/adidas/4.Adidas-Logo.jpg')

    data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    list_image = os.listdir(data)
    for image in list_image:
        # img = cv2.imread(os.path.join(data, image))
        # img = img / 255
        # img = torch.from_numpy(img).float()
        # img = img.permute(2, 0, 1)
        # img = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img) 
        # img = img.to(device).unsqueeze_(0)
        img = Image.open(os.path.join(data, image))
        # print(img.size)
        img = data_transform(img).unsqueeze(0).to(device)
        # print(img.shape)
        # img = torchvision.transforms.ToTensor()(img).to(device)
        # img = torch.tensor(img).unsqueeze(0).to(device, dtype=torch.float)
        # img = data_transform(img)
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
        print(preds)

    for inputs, labels in dataloaders['val']:
        inputs = inputs.to(device)
        # print(inputs[2])
        # sample = torch.randn(1, 3, 224, 224)
        # inp = inp[np.newaxis, :]
        inp = inputs[0]
        inp = inp.unsqueeze(0)
        # inp = torch.cat((inp, sample), 0)        

        # inp = inp.to(device)
        outputs = model(inputs[0:2])
        _, preds = torch.max(outputs, 1)
        # import pdb; pdb.set_trace()
        print(preds, labels[0:2])
    return preds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', help='train or evalute')
    opt = parser.parse_args()
    
    data_dir = '/home/vietthangtik15/cv/dataset'
    dataloaders, dataset_sizes, class_names, device, test_dataloader = data_loader(data_dir)
    print(class_names)

    # for inputs, labels in dataloaders['train']:
    #     print(labels)
    # import pdb; pdb.set_trace()
    model_ft = models.resnet18(pretrained=True)
    for param in model_ft.parameters():
        param.requires_grad = False
    num_ftrs = model_ft.fc.in_features

    model_ft.fc = nn.Linear(num_ftrs, 8)
    # model_ft = nn.Sequential(model_ft, nn.Softmax(1))
    model_ft = model_ft.to(device)
    if opt.mode == 'train':
        model = main_train(model_ft, dataloaders, device, 75)
        for param in model.parameters():
            param.requires_grad = True
        model = main_train(model, dataloaders, device, 100)
    else:
        if opt.mode == 'evalute':
            pred =  evaluate(model_ft, 'checkpoint.pth', dataloaders, device, test_dataloader) 
            print(class_names)
            print(pred)