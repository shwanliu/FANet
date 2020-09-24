import torch
from PIL import Image
# from torchvision import transforms
import torchvision.transforms as T
device = torch.device('cuda')
transform=T.Compose([
                    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])
def prediect(img_path, modelPath, value):
    net=torch.load(modelPath)
    net=net.to(device)
    torch.no_grad()
    img=Image.open(img_path)
    img=transform(img).unsqueeze(0)
    img_ = img.to(device)
    outputs = net(img_)
    zero = torch.zeros_like(outputs.data)
    one = torch.ones_like(outputs.data)
    predicted = torch.where(outputs.data > value, one, zero)
    # _, predicted = torch.max(outputs, 1)
    # print(predicted)
    print('this picture maybe :',classes[predicted[0]])
if __name__ == '__main__':
    f = open('datasets/Anno/class.txt')
    className = []
    lines = f.readlines()
    for line in lines:
        className.append( line.strip().split("：")[1])
    # print(className)
    # modelPath = ""
    # prediect('./test/name.jpg',modelPath，0.6)
