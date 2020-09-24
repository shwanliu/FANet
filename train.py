# encoding=utf-8
from dataReader.celebaData import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
import models
import time
import signal
import torch.nn as nn
from torch.optim import lr_scheduler
from utils import *
from tqdm import tqdm
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
version =  torch.__version__
#os.environ['CUDA_VISIBLE_DEVICES'] = "1"

y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []

y_acc = {} # acc history
y_acc['train'] = []
y_acc['val'] = []

def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./checkpoint',opt.model,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(gpu_ids[0])

def train(**kwargs):
    """
    训练函数
    """
    opt.parse(**kwargs)
    global isTer
    isTer = False  # 设置全局变量方便中断时存储model参数

    # 训练数据集
    dataSets = celebaData(opt.trainFolder,opt.labelFolder)
    # trainLoader = DataLoader(trainData, batch_size=opt.batchSize,shuffle=True, num_workers=opt.numWorker)
    # # 验证集
    # cvData = celebaData(opt.trainFolder,self.labelFolder,isTrain=False, isCV=False, )
    # cvLoader = DataLoader(cvData, batch_size=opt.batchSize,
    #                       shuffle=True, num_workers=opt.numWorker)
    trainNum = len(dataSets) # 训练图像总数
    trainLoader, cvLoader = split_dataset(dataSets, opt.batchSize)
    # 生成模型,使用预训练
    model = eval('models.' + opt.model + '(numClass=' + str(opt.numClass) + ')')
    model.train(True)  # 设置模型为训练模式（dropout等均生效）
    criterion = eval('nn.' + opt.lossFunc + '()')
    # 初始化优化器,要使用不同的学习率进行优化
    # acclerateParams = list(map(id, model.model.fc.parameters())) + list(map(id, model.classifierLayer.parameters()))
    # baseParams = filter(lambda p: id(p) not in acclerateParams, model.parameters())
    # 观测多个变量
    optimizer = torch.optim.SGD(model.parameters(),lr=opt.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    # optimizer = torch.optim.SGD([
    #     {'params': baseParams, 'lr': opt.lr},
    #     {'params': model.model.fc.parameters(), 'lr': 0.1},
    #     {'params': model.class_0.parameters(), 'lr': 0.1},
    #      {'params': model.class_0.parameters(), 'lr': 0.1},
        # {'params': model.classifierLayer.parameters(), 'lr': 0.1}
    # ], momentum=0.9, weight_decay=5e-4, nesterov=True)
    timerOp = lr_scheduler.StepLR(optimizer, step_size=opt.lrDecayRate, gamma=opt.lrDecay)  # 每经过40轮迭代，学习率就变为原来的0.1
    lossVal = []  # 记录损失函数变化
    trainAcc = []  # 记录训练集精度变化
    cvAcc = []  # 记录交叉验证数据集精度变化

    if opt.useGpu:
        model = model.cuda()
    # 开始训练
    signal.signal(signal.SIGINT, sigTerSave)  # 设置监听器方便随时中断
    dataloaders = {'train':trainLoader,'val':cvLoader}
    since = time.time()
    warm_up = 0.1 # We start from the 0.1*lrRate
    warm_iteration = round(len(dataloaders['train'])/opt.batchSize)*opt.warm_epoch # first 5 epoch
    for epoch in range(opt.maxEpoch):
        print('Epoch {}/{}'.format(epoch, opt.maxEpoch - 1))
        print('-' * 10)
        #optimizer.step()
                # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                # scheduler.step()
                timerOp.step()  # 仅有达到40轮之后学习率才会下降
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            # Iterate over data.
            for count, (inputs, labels) in enumerate(tqdm(dataloaders[phase])):
                now_batch_size,c,h,w = inputs.shape
                if now_batch_size<opt.batchSize: # skip the last batch
                    continue
                if opt.useGpu:
                    inputs = Variable(inputs.cuda().detach())
                    labels = Variable(labels.cuda().detach())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # forward
                if phase == 'val':
                    with torch.no_grad():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)

                loss = criterion(outputs, labels.float())
                if epoch<opt.warm_epoch and phase == 'train': 
                    warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                    loss *= warm_up

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                if count!=0 and count % opt.printFreq == 0:
                    print('Its '+phase+' epoch: '+str(epoch) +', step:'+str(count) +' in epoch, ' +phase+'_loss: '+str(loss.item())+' ,'+phase+'_acc: '+str(float(torch.sum(preds == labels.data.float()))/(opt.batchSize*opt.numClass))+', lr: '+str(optimizer.param_groups[0]['lr']))
               
                # statistics
                if int(version[0])>0 or int(version[2]) > 3: # for the new version like 0.4.0, 0.5.0 and 1.0.0
                    # running_loss += loss.item() * now_batch_size
                    running_loss += loss.item()
                else :  # for the old version like 0.3.0 and 0.3.1
                    # running_loss += loss.data[0] * now_batch_size
                    running_loss += loss.data[0]

                zero = torch.zeros_like(outputs.data)
                one = torch.ones_like(outputs.data)
                preds = torch.where(outputs.data > 0.6, one, zero)

                running_corrects += float(torch.sum(preds == labels.data.float()))

            epoch_loss = running_loss / (len(dataloaders[phase]))
            epoch_acc = running_corrects / (len(dataloaders[phase]))
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            y_loss[phase].append(epoch_loss)
            y_acc[phase].append(epoch_acc)  
            
            if phase == 'val':
                time_elapsed = time.time() - since
                print('Training complete in {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))
                print()

                checkpoint = {"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch}
                # save_model_file = os.path.join(args.save_model, 'net_%s.pkl'%epoch)
                model.save('checkpoint/epoch%s'%epoch + opt.model + '.pth')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    np.savetxt("tarin_Acc.txt",  y_acc['train'])
    np.savetxt("val_Acc.txt", y_acc['val'])
    np.savetxt("train_Loss.txt", y_loss['train'])
    np.savetxt("val_Loss.txt", y_loss['val'])
    print('done')
    
def test(**kwargs):
    # 针对test（gallery）数据集进行
    opt.parse(**kwargs)
    # 进行测试，计算相似度
    model = eval('models.' + opt.model + '(' + str(opt.numClass) + ')')
    model.load(opt.modelPath)
    model.res.fc = nn.Sequential()
    model.classifierLayer = nn.Sequential()  # 注意这里不能直接删除，免得前传的时候找不到层，置空就好
    model = model.eval()  # 设置为测试模式，dropout等均失效    
    # 准备数据
    testData = dataReder(opt.testFolder, isTrain=False)
    # 不能洗牌
    testLoader = DataLoader(
        testData, batch_size=opt.batchSize, num_workers=opt.numWorker)
    if opt.useGpu:
        model = model.cuda()
    features = torch.FloatTensor()  # 初始化一个floattensor用于存储特征
    for ii, (imgData, label) in enumerate(testLoader):
        n, _, _, _ = imgData.size()  # 获得图像数目
        doubleF = torch.FloatTensor(n, model.numFin).zero_()  # 存储反转前后提取特征的融合特征
        for jj in range(2):
            inImg = hozFilp(imgData)  # 第一次要反转
            if opt.useGpu:
                inImg = inImg.cuda()
            calF = model(Variable(inImg))  # 提取出2048维特征
            doubleF += calF.data.cpu()  # 融合即是相加
        # feature归一化
        normF = torch.norm(doubleF, p=2, dim=1, keepdim=True)
        doubleF = doubleF.div(normF.expand_as(doubleF))
        features = torch.cat((features, doubleF), 0)  # 将得到的特征按照垂直方向进行拼接
    torch.save(features, "snapshots/allF.pth")
    print("TEST所有特征已经保存")


def val(model, loader):
    # 交叉验证
    acc = []
    criterion = eval('nn.' + opt.lossFunc)
    for ii, (data, label) in enumerate(loader):
        data = Variable(data)
        label = Variable(label)
        if opt.useGpu:
            data = data.cuda()
            label = label.cuda()
        # 进行验证
        score = model(data)
        acc.append(calScore(score, label))
    return torch.mean(torch.FloatTensor(acc))
######################################################################
# Draw Curve
#---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_acc['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_acc['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig( os.path.join('./figure',opt.model,'train.jpg'))

def sigTerSave(sigNum, frame):
    """
    使用ctrl+C时，将模型参数存储下来再退出，这是一个槽
    """
    global isTer
    isTer = True  # 全局变量设置为True
    print('保存模型参数至当前目录的temp.pth...')

if __name__ == '__main__':
    train()
    # import fire

    # fire.Fire()
