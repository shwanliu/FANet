# encoding=utf-8
import matplotlib
matplotlib.use('agg')
import numpy as np
import matplotlib.pyplot as plt

train_loss = []
train_acc = []
val_loss = []
val_acc = []

def ReadLog(logfile):
    f  = open(logfile,'r')
    lines = f.readlines()
    for line in lines:
        if "train Loss" in line:
            acc = line.split('Acc:')[1].strip()
            loss = line.split('Loss: ')[1].split('Acc: ')[0].strip()
            train_loss.append(loss)
            train_acc.append(acc)
        if "val Loss" in line:
            acc = line.split('Acc:')[1].strip()
            loss = line.split('Loss: ')[1].split('Acc: ')[0].strip()
            val_loss.append(loss)
            val_acc.append(acc)
            # print(line)

    return train_loss, train_acc, val_loss, val_acc
train_loss, train_acc, val_loss, val_acc=ReadLog("train0928.log")
x0 =  [i for i in range(len(train_acc))]
y00 = list(map(float,train_acc[0:len(train_acc)]))
y01 = list(map(float,val_acc[0:len(val_acc)]))

y02 = list(map(float,train_loss[0:len(train_loss)]))
y03 = list(map(float,val_loss[0:len(val_loss)]))
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
plt.savefig('train0928.jpg')
# print(lines)