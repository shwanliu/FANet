import torch
import models
from dataReader.celebaData import *
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
import warnings
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
    dataLoader = data.DataLoader(dataSets, batch_size=2,num_workers=8)
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
        if now_batch_size<2: # skip the last batch
            continue
        inputs = Variable(inputs.cuda().detach())
        labels = Variable(labels.cuda().detach())
        outputs = model(inputs)
        criterion = nn.BCELoss()
        loss = criterion(outputs, labels.float())
        zero = torch.zeros_like(outputs.data)
        one = torch.ones_like(outputs.data)
        preds = torch.where(outputs.data > 0.6, one, zero)
        if count!=0 and count % 50== 0:
            print('Its '+phase+' epoch: '+str(epoch) +', step:'+str(count) +' in epoch, ' +phase+'_loss: '+str(loss.item())+' ,'+phase+'_acc: '+str(float(torch.sum(preds == labels.data.float()))/(2*len(classes))))
        
        running_loss += loss.item()
        running_corrects += float(torch.sum(preds == labels.data.float()))
        y_loss.append(epoch_loss)
        y_acc.append(epoch_acc) 
    np.savetxt("train_Loss.txt", y_loss)
    np.savetxt("val_Loss.txt", y_loss)
                
    # draw_table = ImageDraw.Draw(im=img)
    # # img.show()
    # img_=transform(img).unsqueeze(0)
    # #img_ = img.to(device)
    # outputs = net(img_)
    # zero = torch.zeros_like(outputs.data)
    # one = torch.ones_like(outputs.data)
    # predicted = torch.where(outputs.data > value, one, zero)
    # pred = dict()
    # print("==="*15)
    # for i in range(len(classes)):
    #     if outputs[0][i].item()>=value:
    #         # pred[classes[i]]=outputs[0][i].item()
    #         score = float('%.4f' %outputs[0][i].item())
    #         print("属性："+classes[i]+"===》score："+str(score))
    # print("==="*15)
    # draw_table.text((100,100),"11",direction=None)
    # img.show()
    # _, predicted = torch.max(outputs, 1)
    # print(predicted)

if __name__ == '__main__':
    f = open('datasets/Anno/class.txt')
    className = []
    lines = f.readlines()
    for line in lines:
        className.append( line.strip().split("：")[1])
    # print(className)
    #modelPath = "/home/shawnliu/workPlace/face_attr/checkpoint/net_6.pkl"
    modelPath = "checkpoint/epoch10FANet.pth"
    imgDir = '/Users/liuxiaoying/Documents/定稿-毕业论文/毕业论文/使用到的数据集/lfw'
    imgTxt = '/Users/liuxiaoying/workplace/CV_Code/FANet/datasets/Anno/lfw_val.txt'
    prediect(imgDir,imgTxt,modelPath,0.6,className)