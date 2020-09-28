import torch
import models
from dataReader.celebaData import *
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
import warnings
import torch.nn as nn
from tqdm import tqdm
from torch.utils import data
warnings.filterwarnings("ignore")
# from torchvision import transforms
import torchvision.transforms as T
device = torch.device('cuda')
transform=T.Compose([
                    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])

def prediect(imgDir, imgTxt, modelPath, value, classes):
    dataSets = celebaData(imgDir, imgTxt)
    # print(len(dataSets))
    dataLoader = data.DataLoader(dataSets, batch_size=4,num_workers=8)
    # print(len(dataLoader))
    net = models.FANet(40)
    net.load_state_dict(torch.load(modelPath, map_location=torch.device('cpu')))
    #net=net.cuda()
    torch.no_grad()
    #print(net)
    net=net.eval()
    # imgPath = os.path.join(imgDir,imgTxt)
    # img=Image.open(imgPath)
    running_loss = 0.0
    running_corrects = 0.0
    y_loss = []
    y_acc = []
    for count, (inputs, labels) in enumerate(tqdm(dataLoader)):
        now_batch_size,c,h,w = inputs.shape
        if now_batch_size<4: # skip the last batch
            continue
        # inputs = Variable(inputs.cuda().detach())
        # labels = Variable(labels.cuda().detach())

        inputs = Variable(inputs)
        labels = Variable(labels)
        outputs = net(inputs)
        criterion = nn.BCELoss()
        loss = criterion(outputs, labels.float())
        zero = torch.zeros_like(outputs.data)
        one = torch.ones_like(outputs.data)
        preds = torch.where(outputs.data > 0.6, one, zero)
        if count!=0 and count % 50== 0:
            print('step:'+str(count) +', loss: '+str(loss.item())+', acc: '+str(float(torch.sum(preds == labels.data.float()))/(4*len(classes))))
        
        running_loss += loss.item()
        running_corrects += float(torch.sum(preds == labels.data.float()))
        # y_loss.append(epoch_loss)
        # y_acc.append(epoch_acc) 
    np.savetxt("lfw_loss.txt", y_loss)
    np.savetxt("lfw_acc.txt", y_acc)

if __name__ == '__main__':
    f = open('datasets/Anno/class.txt')
    className = []
    lines = f.readlines()
    for line in lines:
        className.append( line.strip().split("：")[1])
    # print(className)
    #modelPath = "/home/shawnliu/workPlace/face_attr/checkpoint/net_6.pkl"
    modelPath = "checkpoint/epoch10FANet.pth"
    imgDir = '/Users/liuxiaoying/Documents/定稿-毕业论文/毕业论文/使用          q到的数据集/lfw'
    imgTxt = '/Users/liuxiaoying/workplace/CV_Code/FANet/datasets/Anno/lfw_val.txt'
    prediect(imgDir,imgTxt,modelPath,0.6,className)