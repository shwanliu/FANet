import torch
import models
from PIL import Image
import warnings
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
def prediect(img_path, modelPath, value, classes):
    net = models.FANet(40)
    net.load_state_dict(torch.load(modelPath, map_location=torch.device('cpu')))
    #net=net.cuda()
    torch.no_grad()
    #print(net)
    net=net.eval()
    img=Image.open(img_path)
    img=transform(img).unsqueeze(0)
    #img_ = img.to(device)
    outputs = net(img)
    zero = torch.zeros_like(outputs.data)
    one = torch.ones_like(outputs.data)
    predicted = torch.where(outputs.data > value, one, zero)
    pred = dict()
    print("==="*10)
    for i in range(len(classes)):
        if outputs[0][i].item()>=value:
            # pred[classes[i]]=outputs[0][i].item()
            score = float('%.4f' %outputs[0][i].item())
            print(classes[i]+" : "+str(score))
    print("==="*10)
    # _, predicted = torch.max(outputs, 1)
    # print(predicted)
if __name__ == '__main__':
    f = open('datasets/Anno/class.txt')
    className = []
    lines = f.readlines()
    for line in lines:
        className.append( line.strip().split("：")[1])
    print(className)
    #modelPath = "/home/shawnliu/workPlace/face_attr/checkpoint/net_6.pkl"
    modelPath = "checkpoint/epoch10FANet.pth"
    prediect('testliu2.jpg',modelPath,0.6,className)
