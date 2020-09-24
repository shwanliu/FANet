import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy
import torchvision
from .BasicNet import *

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(BasicNet):
    def __init__(self, input_dim, class_num=1, activ='sigmoid', num_bottleneck=512):
        super(ClassBlock, self).__init__()

        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block += [nn.LeakyReLU(0.1)]
        add_block += [nn.Dropout(p=0.5)]

        add_block = nn.Sequential(*add_block)
        add_block.apply(self.weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        if activ == 'sigmoid':
            classifier += [nn.Sigmoid()]
        elif activ == 'softmax':
            classifier += [nn.Softmax()]
        elif activ == 'none':
            classifier += []
        else:
            raise AssertionError("Unsupported activation: {}".format(activ))
        classifier = nn.Sequential(*classifier)
        classifier.apply(self.weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x

class FANet(BasicNet):
    """Some Information about MyModule"""
    def __init__(self, numClass):
        super(FANet, self).__init__()
        self.numClass = numClass
        self.model = torchvision.models.resnet50(pretrained=True)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model.fc.apply(self.weights_init_kaiming)  # 初始化权重

        for c in range(numClass):
            self.__setattr__('class_%d' % c, ClassBlock(input_dim=self.numBottleNeck, class_num=1, activ='sigmoid') )

        # 将特征映射到对应类别
        # self.classifierLayer = nn.Sequential(
        #     nn.Linear(self.numBottleNeck, numClass)
        # )
        # self.classifierLayer.apply(self.weights_init_classifier)  # 初始化权重

    def forward(self, x):
        x = self.model(x)
        # print(self.numFin)
        # x = x.squeeze()
        x = x.view(x.size(0), -1)
        # print(x.size())
        output = [self.__getattr__('class_%d' % c)(x) for c in range(self.numClass)]
        # output = torch.cat(output, dim=1)
        return torch.cat(output, dim=1)

if __name__ == "__main__":
    model =  FANet(40)
    testTensor = torch.Tensor(2,3,224,224);
    # print(testTensor)
    # print(model.parameters().class_0)
    print(model(testTensor))