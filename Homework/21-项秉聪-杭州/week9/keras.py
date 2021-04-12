# -*- encoding=UTF-8 -*-
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models,layers
from tensorflow.keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

network = models.Sequential()
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))

network.compile(optimizer='RMSprop',loss='categorical_crossentropy',metrics=['accuracy'])

#将网络打扁，形成1*784
print(train_images.shape)
print(test_images.shape)
train_images = train_images.reshape(train_images.shape[0],train_images.shape[1]*train_images.shape[2])
test_images = test_images.reshape(test_images.shape[0],test_images.shape[1]*test_images.shape[2])
print(train_images.shape)
print(test_images.shape)

#数据归一化到[0，1]
train_images = train_images.astype("float32")/255
test_images = test_images.astype("float32")/255

#label转成一维数组
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#开始训练
network.fit(train_images,train_labels,batch_size=100,epochs=5,verbose=1)

#验证测试数据集
test_acc,test_loss = network.evaluate(test_images,test_labels,batch_size=100,verbose=1)
print("test_acc",test_acc)
print("test_loss",test_loss)

#拿其它数据作推理，其它数据的像素大小与训练数据一致
import cv2
import numpy as np
count = 0
predict_img = range(10)
for i in range(len(predict_img)):
    img_tmp = cv2.imread("images/num" + str(i) + ".png")
    img_tmp = cv2.cvtColor(img_tmp,cv2.COLOR_BGR2GRAY)
    img_tmp = img_tmp.astype("float32")/255
    img_tmp = img_tmp.reshape(1,(img_tmp.shape[0]*img_tmp.shape[1]))
    result = network.predict(img_tmp,verbose=0)
    print("result",result)
    result_idx = np.nanargmax(result[0])
    print("predict_result:",result_idx," real_result:",i)
    if result_idx == i:
        count += 1
    print("accuracy:{0}%".format(count*100/(i+1)))
#奇怪了， 自己画的图测算出来正确率简直是可怜




