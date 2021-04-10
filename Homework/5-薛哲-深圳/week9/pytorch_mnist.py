# 多分类问题用到softmax分类器,希望输出之间是带有竞争抑制作用的，即输出是一个概率分布
# 输出层变成softmax层，满足输出条件（每一个输出大于等于0，且和为1：先对输入求指数然后除以所有数的和）
'''
import numpy as np
y = np.array([1, 0, 0])
z = np.array([0.2, 0.1, -0.1])
y_pred = np.exp(z)/np.exp(z).sum()  # softmax
loss = (-y*np.log(y_pred)).sum()  # NLLLoss
softmax+log+NLLLoss就是pytorch里面提供的torch.nn.CrossEntropyLoss()交叉熵损失，注意softmax前面一层不需要做激活
并且y必须得是torch.LongTensor([...])
'''
# 经过softmax后的Y_hat再取log，然后与Y_true的标签负值相乘，这就是Negative log likelihood loss（NLLLoss）函数
# 手写数字识别的多分类问题

import torch

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets


# 准备数据
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])   #  Normalize(mean,std)
# 数据的预处理，ToTensor将单通道图片（灰度图片）转换成1*W*H的张量，或者是多通道图片（彩色三通道RGB）转换成3*W*H的张量
# Normalize将数据归一化成0-1分布：pixel_norm = (pixel_origin - mean)/std ;mean为数据集的像素值均值，std为标准差(两者均为经验或者计算所得)


train_dataset = datasets.MNIST(root='../dataset/mnist', train=True, transform= transform, download=True)
test_dataset = datasets.MNIST(root='../dataset/mnist', train=False, transform= transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 定义模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(784, 512)
        self.linear2 = torch.nn.Linear(512, 256)
        self.linear3 = torch.nn.Linear(256, 128)
        self.linear4 = torch.nn.Linear(128, 64)
        self.linear5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        x = torch.relu(self.linear4(x))
        x = self.linear5(x)  # 最后一层后不需要做非线性处理
        return x


model = Model()

# 定义loss和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # 冲量是0.5，给数据添加惯性防止得到局部极值

def train(epoch):   #  把一轮训练封装成函数
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        # 数据集除以batch_size得到batch的个数，batch_idx表示第几个batch，由enumerate函数返回
        inputs, target = data
        optimizer.zero_grad()

        # forward + backword + update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 ==299:  # 每三百个batch输出一回loss
            print('[epoch=%d, batch=%5d] loss: %.3f' %(epoch+1, batch_idx+1, running_loss/300))
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            # torch.max()函数返回最大值和最大值的下标，下标对应预测的类别;outputs.data是一个N*10的tensor,N是batch_size
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # predicted == labels真就是1，假就是0
    print('Accuracy on test set: %d %%' % (100*correct/total))  # 返回准确率

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()