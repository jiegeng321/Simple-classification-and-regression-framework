import numpy as np
#import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torch import Tensor
import os
import random
import math
class DatasetImgTxtPair(Dataset):

    def __init__(self, root, train_val="train", transform = None, target_transform=None):
        super(DatasetImgTxtPair, self).__init__()
        self.train_val = train_val
        self.transform = transform
        self.target_transform = target_transform

        if self.train_val == "train" :
            #file_annotation = root + '/annotations/cifar10_train.json'
            self.img_folder = root + '/train/images'
            self.txt_folder = root + '/train/labels'
        else:
            self.img_folder = root + '/val/images'
            self.txt_folder = root + '/val/labels'

        self.imgnames = sorted(os.listdir(self.img_folder))
        self.labels = sorted(os.listdir(self.txt_folder))
        assert len(self.imgnames) == len(self.labels)


    def __getitem__(self, index):
        img_name = self.img_folder + '/' + self.imgnames[index]
        label_name = self.txt_folder + '/' + self.imgnames[index] + ".dat.txt"
        if not os.path.exists(label_name):
            label_name = self.txt_folder + '/' + self.imgnames[index][:-4] + ".txt"
        #assert img_name.split(".")[0] == label_name.split(".")[0]
        img = Image.open(img_name)
        wd, ht = img.size
        #img = img.convert('L')
        def Srotation_angle_get_coor_coordinates(point, center, angle):
            src_x, src_y = point
            center_x, center_y = center
            radian = math.radians(angle)
            dest_x = round((src_x - center_x) * round(math.cos(radian),15) + (src_y - center_y) * round(math.sin(radian),15) + center_x)
            dest_y = round((src_y - center_y) * round(math.cos(radian),15) - (src_x - center_x) * round(math.sin(radian),15) + center_y)
            return int(dest_x), int(dest_y)
        def random_rotate(min_rate=0, max_rate=25):
            rotate_rate = random.randint(min_rate, max_rate)
            if random.randint(0,1):
                rotate_rate *= -1
            return rotate_rate
        angle = 0
        if random.random() < 0.8:
            angle = random_rotate(0,45)
        
        with open(label_name,"r") as f:
            lines = f.read()
            #label = [float(i)*144.0/wd/144.0 if n % 2 ==0 else float(i)*144.0/ht/144.0 for n,i in enumerate(lines.strip(' ').split(' '))]
            #print(label_name)
            #print(lines)
            label = [float(i)  for i in lines.strip(' ').split(' ')]
            #for l in label:
            #    print(l)
            #    print(float(l))
        if angle !=0:
            label_rot = []
            img = img.rotate(angle,expand=0)
            for i in range(22):
                x,y=Srotation_angle_get_coor_coordinates((label[2*i],label[2*i+1]),(wd/2,ht/2),angle)
                np.clip(x,0,wd-1)
                np.clip(y,0,ht-1) 
                label_rot.append(x)
                label_rot.append(y)
        if angle !=0:
            label = [float(i)*144.0/wd if n % 2 ==0 else float(i)*144.0/ht for n,i in enumerate(label_rot)]
        else:
            label = [float(i)*144.0/wd if n % 2 ==0 else float(i)*144.0/ht for n,i in enumerate(label)]
        '''
        img = img.resize((144,144))
        imagedraw = ImageDraw.Draw(img)
        imagedraw.line([(int(label[0]),int(label[1])),(int(label[2]),int(label[3])),(int(label[4]),int(label[5])),(int(label[6]),int(label[7])),(int(label[8]),int(label[9]))],(255,0,0),width=5)
        imagedraw.line([(int(label[10]),int(label[11])),(int(label[12]),int(label[13])),(int(label[14]),int(label[15])),(int(label[16]),int(label[17])),(int(label[18]),int(label[19]))],(255,0,0),width=5)
        imagedraw.line([(int(label[20]),int(label[21])),(int(label[22]),int(label[23])),(int(label[24]),int(label[25])),(int(label[26]),int(label[27])),(int(label[28]),int(label[29])),(int(label[30]),int(label[31]))],(255,0,0),width=5)
        imagedraw.line([(int(label[32]),int(label[33])),(int(label[34]),int(label[35])),(int(label[36]),int(label[37])),(int(label[38]),int(label[39])),(int(label[40]),int(label[41])),(int(label[42]),int(label[43]))],(255,0,0),width=5)
        img.save('./test/'+self.imgnames[index])
        '''
        if self.transform is not None:
            img = self.transform(img)
        
        label = Tensor(label)
        return img, label

    def __len__(self):
        return len(self.imgnames)
