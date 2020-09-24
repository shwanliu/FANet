# -*-coding:utf-8-*-
from PIL import Image
import torchvision.transforms as T
from torch.utils import data
import os
import sys
import numpy as np
from collections import defaultdict

class celebaData(data.Dataset):
    """Some Information about celebaData"""
    def __init__(self,baseDir,labelPath, transformation=None, isTrain=True, isCV=False):
        self.data_list = []
        # self.img_transform = img_transform
        with open(labelPath,'r') as f:
            lines = f.readlines()
        for line in lines:
            if '.jpg' in line:
                line = line.strip().split(' ')
                data_dic = {}
                # 构造一个字典，每一个字典都有相应的数据读取字段，mtcnn包括图片，以及label，bbox，landmark，都进行初始化，避免不存在的label，
                # 例如landmar不存在，bbox不存在的情况，这个时候将他置为你np.zeros
                data_dic['image'] = os.path.join(baseDir,line[0])
                tmpLabel = []
                for value in line[1:]:
                    if value == "1":
                        tmpLabel.append(1)
                    if value == "-1":
                        tmpLabel.append(0)
                # print(len(data_dic['label']))
                data_dic['label']=tmpLabel
                self.data_list.append(data_dic)
         # 给出默认变换
        if transformation is None:
            # 给一个默认的变换
            if isTrain:
                # 训练数据集要使用特殊变换
                self.trans = T.Compose([
                    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
                    T.Resize((288, 288)),
                    T.RandomCrop((256, 256)),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                self.trans = T.Compose([
                    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
                    T.Resize((256, 256)),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

    def __getitem__(self, index):
        data = self.data_list[index]
        # print(data['image'])
        data['image'] = self.trans(Image.open(data['image']))
        data['label'] = np.array(data['label'])
        return data

    def __len__(self):
        return len(self.data_list)

def split_dataset(dataset, batch_size):
    data_size = len(dataset)
    validation_split = .2
    shuffle = True
    random_seed = 42

    indices = list(range(data_size))
    #print(validation_split * data_size)
    split = int(np.floor(validation_split * data_size))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sample = data.SubsetRandomSampler(train_indices)
    validation_sample = data.SubsetRandomSampler(val_indices)

    train_loader = data.DataLoader(dataset, batch_size=batch_size, sampler=train_sample, num_workers=4)
    val_loader = data.DataLoader(dataset, batch_size=batch_size, sampler=validation_sample, num_workers=4)

    return train_loader, val_loader

if __name__ == "__main__":
    dataset =  celebaData("/Volumes/TOSHIBA EXT/CelebA/data/img_align_celeba","../datasets/Anno/list_attr_celeba.txt")
    print(dataset[0])
    print(type(dataset[0]['label']))