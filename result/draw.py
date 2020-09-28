# encoding=utf-8
import matplotlib
import numpy as np
matplotlib.use('agg')
import matplotlib.pyplot as plt

train_Acc=open('tarin_Acc.txt','r')
train_Loss=open('train_Loss.txt','r')
val_Acc=open('val_Acc.txt','r')
val_Loss=open('val_Loss.txt','r')

def ReadTxtName(rootdir):
    lines = []
    with open(rootdir, 'r') as file_to_read:
        while True:
            line = file_to_read.readline()
            if not line:
                break
            if "Acc" in rootdir:
                line = float(line.strip('\n'))/3200
            lines.append(str(line))
    return lines

lineslist1=ReadTxtName('tarin_Acc.txt')
lineslist2=ReadTxtName('val_Acc.txt')
lineslist3=ReadTxtName('train_Loss.txt')
lineslist4=ReadTxtName('val_Loss.txt')

x0 =  [i for i in range(len(lineslist1))]
y00 = list(map(float,lineslist1[0:len(lineslist1)]))
y01 = list(map(float,lineslist2[0:len(lineslist2)]))

y02 = list(map(float,lineslist3[0:len(lineslist3)]))
y03 = list(map(float,lineslist4[0:len(lineslist4)]))
# print(len(x0))
# print(y03)
# print(y02)
fig = plt.figure(figsize = (12,8)) 
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.title('Epochs & Train Loss',fontsize=18)
# plt.subplot(1,2,1) #要生成两行两列，这是第一个图plt.subplot('行','列','编号')
plt.plot(x0, y00, '.-', label='tarin_Acc')
plt.plot(x0, y01, '.-', label='val_Acc')
# plt.grid(True)
# plt.subplot(1,2,2) #两行两列,这是第二个图
plt.plot(x0, y02, '.-', label='train_Loss')
plt.plot(x0, y03, '.-', label='val_Loss')
plt.grid(True)
plt.legend(prop = {'size':16})
plt.xticks(np.arange(0.0, 110, step=5))
plt.yticks(np.arange(0.0, 1.1, step=0.05))
plt.xlabel('Epochs',fontsize=16)
# plt.ylabel('Train Loss',fontsize=16)
plt.tick_params(labelsize=16)
#plt.legend() # 将样例显示出来
plt.show()
plt.savefig('testblueline.jpg')