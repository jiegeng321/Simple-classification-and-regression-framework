from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from models import initialize_model
from tools import check_dir
from datasets import DatasetImgTxtPair
from losses import AdaptiveWingLoss, WingLoss, AWing
from tqdm import tqdm
from torch import Tensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
#data_dir = "/home/admin/data/xfeng/COMDATA/face_pts_train_data/face_pts_train_data_torch_comb_6parts"
data_dir = "../comdata/insurance_angle_train_data"
save_model_dir = '../commodel/tmp_class2/'

model_name = "efficientnet"  # 从[my_resnet,resnet, alexnet, vgg, squeezenet, densenet, inception,efficientnet]中选择模型
pretrain_model = None#"/home/admin/data/xfeng/COMMODEL/face_pts_20th_with_init_final_no_open_small2/best_acc.pt"
layers = [1,1,1,1,1]
input_size = 224
num_classes = 4
batch_size = 32
num_epochs = 500
lr = 0.01
lr_sch = [num_epochs//2,int(num_epochs*7/8)]
gamma = 0.1
feature_extract = False

criterion = nn.CrossEntropyLoss()
#criterion = AdaptiveWingLoss()
#criterion = nn.MSELoss()
#criterion = WingLoss()
#criterion = AWing()

data_transforms = {
    'train': transforms.Compose([
        #transforms.Grayscale(num_output_channels=1),
        transforms.Resize((input_size,input_size)),
        #transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)],p=0.5),
        #transforms.RandomResizedCrop(input_size),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ]),
    'val': transforms.Compose([
        #transforms.Grayscale(num_output_channels=1),
        transforms.Resize((input_size,input_size)),
        #transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ]),
}

check_dir(save_model_dir)
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 1000000
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 每个epoch都有一个训练和验证阶段
        for phase in ['train', 'val']:
            start_time = time.time()
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # 迭代数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计
                scheduler.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            #print(len(dataloaders[phase].dataset))
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            train_time = time.time()-start_time
            print('{} Loss: {:.4f} Acc: {:.4f} Time cost:{:.2f}s'.format(phase, epoch_loss, epoch_acc, train_time))

            # deep copy the model

            #if phase == 'val':
               #torch.save(model, save_model_dir+'model_epoch_%d'%epoch+'.pth')
            if (phase == 'val' and epoch_loss < best_loss):#epoch_acc > best_acc
                best_loss = epoch_loss
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), os.path.join(save_model_dir,'best_acc.pt'))
                #torch.save(model, os.path.join(save_model_dir,'best_acc.pt'))
                print("better model is saved in %s"%(os.path.join(save_model_dir,'best_acc.pt')))
            if phase == 'val':
                val_acc_history.append(epoch_acc)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

if __name__ == '__main__':
    # 在这步中初始化模型
    model_ft = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True, pretrain_model=pretrain_model)

    # 打印我们刚刚实例化的模型
    print(model_ft)

    print("Initializing Datasets and Dataloaders...")

    # 创建训练和验证数据集
    #image_datasets = {x: DatasetImgTxtPair(data_dir, train_val=x, transform=data_transforms[x]) for x in ['train', 'val']}
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    print(image_datasets["train"].class_to_idx)
    #print(image_datasets["train"].imgs)
    # 创建训练和验证数据加载器
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}


    # 将模型发送到GPU
    model_ft = model_ft.to(device)

    # 在此运行中收集要优化/更新的参数。
    # 如果我们正在进行微调，我们将更新所有参数。
    # 但如果我们正在进行特征提取方法，我们只会更新刚刚初始化的参数，即`requires_grad`的参数为True。
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)
    optimizer_ft = optim.Adam(params_to_update, lr=lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=lr_sch, gamma=gamma)

    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, scheduler, num_epochs=num_epochs, is_inception=(model_name=="inception"))

