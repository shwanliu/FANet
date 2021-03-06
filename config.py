# -*-coding:utf-8-*-
import warnings


class Config(object):
    def __init__(self):
        """
        初始化系列参数方便修改
        """
        # 这一部分设置数据导入导出路径
        self.trainFolder = '/home/shawnliu/workPlace/FANet/datasets/img/'  # 训练图像路径
        self.labelFolder = './datasets/Anno/list_attr_celeba.txt'  # 训练图像路径
        self.testFolder = 'dataReader/test/'  # 测试图像路径
        self.modelPath = 'checkpoint/'  # 模型存储路径
        # 这一部分设置模型基本参数
        self.lr = 0.01
        self.warm_epoch = 0
        self.weightDecay = 1e-3
        self.batchSize = 96
        self.maxEpoch = 100
        self.lrDecay = 0.1  # 学习率下降率
        self.lrDecayRate = 40  # 每隔那么多次，学习率就下降一下
        self.numClass = 40
        self.lossFunc = 'BCEWithLogitsLoss'  # 对应的损失函数
        # self.lossFunc = 'CrossEntropyLoss'  # 对应的损失函数
        # 这一部分设置模型训练中其余的参数
        self.printFreq = 80  # 隔80次打印一下
        self.useGpu = True  # 要不要GPU
        self.model = 'FANet'  # 模型名字
        self.numWorker = 4  # 线程数
        self.snapFreq = 1000  # 每过这么多轮保存一下快照
        self.topN = 6  # 计算CMC曲线排名前几

    def parse(self, **kwargs):
        # 按照用户配置更新参数
        for k, v in kwargs.items():
            if not getattr(self, k):
                warnings.warn("no %s found" % k)
            setattr(self, k, v)
        print('user config:')
        for k, v in vars(self).items():
            if not k.startswith('__'):
                # 非内部构建的函数
                print(k, v)


opt = Config()  # 实例化一个对象好导出
