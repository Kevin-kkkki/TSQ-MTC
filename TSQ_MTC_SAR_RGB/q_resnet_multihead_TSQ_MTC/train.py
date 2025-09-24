import copy
import os

import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import numpy as np

from datasets import CustomDataset, data_transforms
from torch.utils.data import random_split

import argparse
from torch.utils.tensorboard import SummaryWriter
from create_mh import MultiInputResNet18
import random


def train_model(model, dataloaders, criterion, optimizer, scheduler, device, writer, num_epochs=50, model_name='best.pth'):
    since = time.time()
    best_acc = 0
    model.to(device)

    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    lrs = []

    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 2 phases:train and valid
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            iter_opt = iter(dataloaders[phase][0])
            iter_sar = iter(dataloaders[phase][1])
            # print(len(dataloaders[phase][0]))
            # print(len(dataloaders[phase][1]))

            loaders = [(iter_opt, 'opt'), (iter_sar, 'sar')]
            
            iteration = 0
            
            while loaders:
                current_iter, label = random.choice(loaders)
                
                try:
                    inputs, labels, _, _ = next(current_iter)
                    # print(f'DataLoader {label}')
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    

                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs, task=0 if label=='opt' else 1)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                        # print(loss)

                        # backward in train
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    iteration += 1
                    
                except StopIteration:
                    loaders = [(it, lbl) for it, lbl in loaders if lbl != label]

            epoch_loss = running_loss / (len(dataloaders[phase][0].dataset) + len(dataloaders[phase][1].dataset))
            epoch_acc = running_corrects.double() / (len(dataloaders[phase][0].dataset) + len(dataloaders[phase][1].dataset))

            time_elapsed = time.time() - since
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # best model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, model_name)
            if phase == 'valid':
                writer.add_scalar('Acc/valid', epoch_acc, epoch)
                writer.add_scalar('Loss/valid', epoch_loss, epoch)
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
            if phase == 'train':
                writer.add_scalar('Acc/train', epoch_acc, epoch)
                writer.add_scalar('Loss/train', epoch_loss, epoch)
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)

        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, lrs

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch ResNet18')
    
    parser.add_argument('--batch_size', type=int, default=32,
            help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,
            help='number of epochs to train (default: 50)')
    parser.add_argument('--lr_epochs', type=int, default=10,
            help='number of epochs to decay the lr (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3,
            help='learning rate (default: 1e-3)')
    parser.add_argument('--seed', type=int, default=23,
            help='random seed (default: 23)')
    parser.add_argument('--data_dir', type=str, default='../dataset/train_ratio_0.5_SAR_False',
            help='data dir (default: ../dataset/train_ratio_0.5_SAR_False)')
    parser.add_argument('--ckpt_path', type=str, default='',
            help='checkpoint path')
    parser.add_argument('--file_name', type=str, default='best.pth',
            help='model dir (default: best.pth)')
    parser.add_argument('--exp_name', type=str, default='exp1',
            help='exp name')

    args = parser.parse_args()

    # trainable_layers = ['layer3', 'layer4']
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    
    # TensorBoard
    writer = SummaryWriter(os.path.join('runs', args.exp_name))

    if not os.path.exists('result'):
        os.mkdir('result')
    model_name = os.path.join('result', args.exp_name + '_' +args.file_name)

    valid_dir = os.path.join(args.data_dir, 'valid')
    train_dir = os.path.join(args.data_dir, 'train')
    train_dataset_opt = CustomDataset(root_dir=train_dir, datatype='RGB', transform=data_transforms['train']) # 12
    print(train_dataset_opt.__len__()) # 1023
    train_dataset_sar = CustomDataset(root_dir=train_dir, datatype='SAR', transform=data_transforms['train']) # 32
    print(train_dataset_sar.__len__()) # 361
    valid_dataset_opt = CustomDataset(root_dir=valid_dir, datatype='RGB', transform=data_transforms['valid'])
    print(valid_dataset_opt.__len__()) # 258
    valid_dataset_sar = CustomDataset(root_dir=valid_dir, datatype='SAR', transform=data_transforms['valid'])
    print(valid_dataset_sar.__len__()) # 91

    train_loader_opt = DataLoader(train_dataset_opt, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train_loader_sar = DataLoader(train_dataset_sar, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader_opt = DataLoader(valid_dataset_opt, batch_size=args.batch_size, shuffle=True)
    valid_loader_sar = DataLoader(valid_dataset_sar, batch_size=args.batch_size, shuffle=True)
    dataloaders = {'train': [train_loader_opt, train_loader_sar], 'valid': [valid_loader_opt, valid_loader_sar]}

    resnet18_base = torchvision.models.resnet34(pretrained=True)
    resnet18 = MultiInputResNet18(resnet18_base, len(train_dataset_opt.categories), len(train_dataset_sar.categories), ckpt_path=args.ckpt_path, quant=True)
    print(resnet18)

    # optimizer and scheduler
    optimizer = optim.Adam(resnet18.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_epochs, gamma=0.1)
    
    criterion = nn.CrossEntropyLoss()

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, val_acc_history, train_acc_history, valid_losses, train_losses, lrs = train_model(model=resnet18,
                                                                                            dataloaders=dataloaders,
                                                                                            criterion=criterion,
                                                                                            optimizer=optimizer,
                                                                                            scheduler=scheduler,
                                                                                            device=device,
                                                                                            writer=writer,
                                                                                            num_epochs=args.epochs, 
                                                                                            model_name=model_name
                                                                                            )


    writer.close()

if __name__ == "__main__":
    main()