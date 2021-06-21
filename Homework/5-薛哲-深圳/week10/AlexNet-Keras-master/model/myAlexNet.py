'''
自写AlexNet-keras框架网络并训练猫狗图片
'''
from keras.models import Sequential  # 导入序列模型类
# 导入layer层的各种layer层
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.datasets import mnist  # 这里训练猫狗两类，不用mnist
from keras.utils import np_utils
from keras.optimizers import Adam  # 导入优化函数,此函数在模型里用不到，在训练里能用到


def AlexNet(input_shape=(224, 224, 3), output_shape=2):
    # 模型实例化
    model = Sequential()
    # 接下来的网络结构根据网络结构图搭建

    # 第一层==========================================
    # 11*11,s=4,out_shape=(55,55,96)
    model.add(
        Conv2D(filters=48, kernel_size=(11, 11), strides=(4, 4), padding='valid', input_shape=input_shape,
               activation='relu')
    )
    # 在每一个卷积层后加个BN层，网络结构图里并没有画出来
    model.add(BatchNormalization())
    # pooling层：3*3，s=2
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    # 第二层==========================================
    # 5*5,s=1,out_shape=(27,27,256)
    model.add(
        Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
    )
    # 在每一个卷积层后加个BN层，网络结构图里并没有画出来
    model.add(BatchNormalization())
    # pooling层：3*3，s=2
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    # 第三层==========================================
    # 3*3,s=1,out_shape=(13,13,384),没有pool，要不要BN看效果
    model.add(
        Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
    )


    # 第4层==========================================
    # 3*3,s=1,out_shape=(13,13,384),没有pool，要不要BN看效果
    model.add(
        Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
    )


    # 第五层==========================================
    # 3*3,s=1,out_shape=(13,13,256),
    model.add(
        Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
    )

    # Max_pool 3*3，s=2, out_shape=(6, 6, 256)
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    # fc层===========================================
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))
    # model.add(Dense(2048, activation='relu'))
    # model.add(Dropout(0.25))
    # model.add(Dense(1024, activation='relu'))
    # model.add(Dropout(0.25))

    # 输出层==========================================
    model.add(Dense(output_shape, activation='softmax'))

    return model
