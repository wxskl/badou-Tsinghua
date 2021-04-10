'''
加载数据
train_images,train_labels表示用来训练的图片以及标签
test_images,test_labels表示用来测试的图片和标签
'''
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 查看图片信息
print('trians shape: ', train_images.shape)  # (60000, 28, 28)
print('trains labels: ', train_labels)
print('test shape: ', test_images.shape)  # (10000, 28, 28)
import matplotlib.pyplot as plt
temp_image = test_images[0]
plt.imshow(temp_image)
plt.show()

# 数据预处理
'''
在把数据输入到网络模型之前，把数据做归一化处理:
1.reshape(60000, 28*28）:train_images数组原来含有60000个元素，每个元素是一个28行，28列的二维数组，
现在把每个二维数组转变为一个含有28*28个元素的一维数组.
2.由于数字图案是一个灰度图，图片中每个像素点值的大小范围在0到255之间.
3.train_images.astype(“float32”)/255 把每个像素点的值从范围0-255转变为范围在0-1之间的浮点值。
'''
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype("float32")/255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype("float32")/255

'''
把图片对应的标记也做一个更改：
目前所有图片的数字图案对应的是0到9。
例如test_images[0]对应的是数字7的手写图案，那么其对应的标记test_labels[0]的值就是7。
我们需要把数值7变成一个含有10个元素的数组，然后在第7个元素设置为1，其他元素设置为0。
例如test_lables[0] 的值由7转变为数组[0,0,0,0,0,0,0,1,0,0,]
'''
from tensorflow.keras.utils import to_categorical
print('before change:', train_labels[0])  # 查看一下标签更改前的值
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print('after change:', train_labels[0])


'''
构建网络模型(两种构建方式，序列构建Sequential add, 通用模型构建：Dense()(input) Model(input,output)
1.layers:表示神经网络中的一个数据处理层。(dense:全连接层)
2.models.Sequential():表示把每一个数据处理层串联起来.
3.layers.Dense(…):构造一个数据处理层。除了Dense还有Input,Activation, Conv2D, MaxPooling2D,Flatten等等
4.input_shape(28*28,):表示当前处理层接收的数据格式必须是长和宽都是28的二维数组，
后面的“,“表示数组里面的每一个元素到底包含多少个数字都没有关系.
'''
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Conv2D, MaxPooling2D

# 方式1：序列构造
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(28*28,)))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 方式2：通用模型构造
'''
x_input = Input(shape=(28, 28,)  # 将输入变成张量
dense_1 = Dense(512,activation='relu')(x_input)
dense_2 = Dense(256,activation='relu')(dense_1)
output = Dense(10,activation='softmax')(dense_2)
model = Model(inputs=x_input, outputs=output)

'''

# 模型优化
model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])  # 优化器及loss函数设定
'''
optimizer：优化器，如：’SGD‘，’Adam‘等。
loss：定义模型的损失函数，如：’mse’，’mae‘等。
metric：模型的评价指标，如：’accuracy‘等。
'''

# 训练
'''
把数据输入网络进行训练：
train_images：用于训练的手写数字图片；
train_labels：对应的是图片的标记；
batch_size：每次网络从输入的图片数组中随机选取128个作为一组进行计算。
epochs:每次计算的循环是五次
'''
model.fit(train_images, train_labels, batch_size=128, epochs=5)

# 测试
'''
测试数据输入，检验网络学习后的图片识别效果.
识别效果与硬件有关（CPU/GPU）.
'''
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
print('test_loss:', test_loss)
print('test_acc:', test_acc)

# 输入一张自己手写的数字图片看看识别效果
import cv2
import numpy as np
img_path = 'mydigit.jpg'
myimg = cv2.imread(img_path)
myimg = cv2.cvtColor(myimg, cv2.COLOR_BGR2GRAY)
myimg = cv2.resize(myimg, (28, 28), interpolation=cv2.INTER_CUBIC)
plt.imshow(myimg)
plt.show()

myimg = np.array(myimg)
myimg = myimg.reshape((1, 28*28))
res = model.predict(myimg)
print(res)
for i in range(10):
    if res[0][i]==1:
        print('这张图片里的数字是：', i)
        break

# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# digit = test_images[1]
# plt.imshow(digit, cmap=plt.cm.binary)
# plt.show()
# test_images = test_images.reshape((10000, 28*28))
# res = model.predict(test_images)
# print(res[1])
# for i in range(res[1].shape[0]):
#     if (res[1][i] == 1):
#         print("the number for the picture is : ", i)
#         break