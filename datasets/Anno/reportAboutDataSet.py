import os
import numpy as np
labelPath="list_attr_celeba.txt"
import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

with open(labelPath,'r') as f:
    lines = f.readlines()
    x = np.zeros((len(lines),40))
    i = 0
    for line in lines:
        if '.jpg' in line:
            line = line.strip().split(' ')
            data_dic = {}
            # 构造一个字典，每一个字典都有相应的数据读取字段，mtcnn包括图片，以及label，bbox，landmark，都进行初始化，避免不存在的label，
            # 例如landmar不存在，bbox不存在的情况，这个时候将他置为你np.zeros
            data_dic['image'] = line[0]
            tmpLabel = []
            line[1:] = [i for i in line[1:] if i != '']
#             print(line[1:])
            for j in range(len(line[1:])):
                if line[1:][j] == "1":
                    x[i][j]=1
                if line[1:][j] == "-1":
                    x[i][j]=0             
            i+=1
            # print(len(data_dic['label']))
#             print(x.shape)

