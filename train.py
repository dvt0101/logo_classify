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
import shutil
from dataloader import *
import argparse
from PIL import Image
from collections import OrderedDict
from sklearn.metrics import confusion_matrix
def model_init(num_class=2):
    # resnet18
    # model_ft = models.resnet18(pretrained=True)
    # num_ftrs = model_ft.fc.in_features
    # model_ft.fc  = nn.Linear(num_ftrs, num_class)

    # mobilenet_v2
    model_ft = models.mobilenet_v2(pretrained=True)
    classifier = nn.Linear(1000, num_class)
    model_ft = nn.Sequential(OrderedDict([
                            ('base', model_ft), 
                            ('fc', classifier)]))
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
                }, 'checkpoint/checkpoint_322.pth')

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
    optimizer_ft = optim.SGD([
                {'params': model.base.parameters()},
                {'params': model.fc.parameters(), 'lr': 1e-3}
            ], lr=1e-2, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model, dataloaders, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=num_epochs)
    return model_ft

def evaluate(model, checkpoint_, dataloader, labels, dataset):
    
    # checkpoint = torch.load(checkpoint_, map_location=torch.device('cpu'))
    checkpoint = torch.load(checkpoint_)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(device)
    accuracy = 0
    y_true, y_pred = [], []
    for inputs, labels in dataloaders[dataset]:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.data)
        y_pred.extend(preds)
        accuracy += torch.sum(preds == labels.data)
    y_true = [x.item() for x in y_true]
    y_pred = [x.item() for x in y_pred]

    confusion = confusion_matrix(y_true, y_pred)
    print(confusion)
    accuracy = accuracy.double()/ dataset_sizes[dataset]
    print('Accuracy: {:2f}'.format(accuracy))
    return accuracy

def predict(model, checkpoint_, dataset = 'val'):

    transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
    checkpoint = torch.load(checkpoint_)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    path = os.path.join('/home/vietthang/ge3f/v2/dataset', dataset)
    result = '/home/vietthang/ge3f/v2/result'
    overkill = []
    underkill = []
    for cat in ['fail', 'pass']:
        data = os.path.join(path, cat)
        images = os.listdir(data)
        print('number of {}: {}'.format(cat, len(images)))
        for name in images:
            img = cv2.imread(os.path.join(data, name))
            # print(img)
            img = torch.from_numpy(img / 255).float().permute(2, 0, 1)
            img = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img) 
            # img = torchvision.transforms.Resize(256)
            img = torch.tensor(img).detach().unsqueeze(0).to(device, dtype=torch.float)
            outputs = model(img)
            _, pred = torch.max(outputs, 1)

            if cat == 'fail':
                if pred.item() == 1:
                    shutil.copy(os.path.join(data, name), os.path.join(result, 'underkill' , dataset))
                    underkill.append(name)
                    # print(name)
            if cat == 'pass':
                if pred.item() == 0:
                    shutil.copy(os.path.join(data, name), os.path.join(result, 'overkill' , dataset))
                    overkill.append(name)
                    # print(name)
    return overkill, underkill

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', help='train or evalute')
    parser.add_argument('--epochs', default=25, type=int, help='Number of epoch')
    opt = parser.parse_args()
    
    data_dir = '/home/vietthang/ge3f/v2/dataset'
    dataloaders, dataset_sizes, class_names = data_loader(data_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    model_ft = model_init()
    # print(model_ft)
    model_ft = model_ft.to(device)
    if opt.mode == 'train':
        model = main_train(model_ft, dataloaders, num_epochs=opt.epochs)

    elif opt.mode == 'evaluate':
            pred =  evaluate(model_ft, 'checkpoint/checkpoint_322.pth', dataloaders, class_names, dataset='val') 
    elif opt.mode == 'predict':
        overkill, underkill = predict(model_ft, 'checkpoint/checkpoint_322.pth', dataset='train')
        print('overkill: {}'.format(len(overkill)))
        print('underkill: {}'.format(len(underkill)))
