from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import np_utils
from keras.optimizers import SGD
from model.myAlexNet import AlexNet
import numpy as np
import utils
import cv2
from keras import backend as K

# K.set_image_dim_ordering('tf')
K.image_data_format() == 'channels_first'


def generate_arrays_from_file(lines, batch_size):
    # 获取数据集的总长度
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            # 从文件中读取图像
            img = cv2.imread(r"E:\pyprogram\dataset\image\train/" + name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255
            X_train.append(img)
            Y_train.append(lines[i].split(';')[1])
            # 读完一个周期后重新开始
            i = (i + 1) % n
        # 处理图像
        X_train = utils.resize_image(X_train, (224, 224))
        X_train = X_train.reshape(-1, 224, 224, 3)
        Y_train = np_utils.to_categorical(np.array(Y_train), num_classes=2)
        yield (X_train, Y_train) # yield相当于return不过下次进入循环从上次退出的地方开始循环，不是从头开始


if __name__ == "__main__":
    # 模型保存的位置
    log_dir = "./logs/"

    # 打开数据集的txt
    with open(r".\data\dataset.txt", "r") as f:
        lines = f.readlines()

    # 打乱行，这个txt主要用于帮助读取数据来训练
    # 打乱的数据更有利于训练
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    # 90%用于训练，10%用于估计。
    num_val = int(len(lines) * 0.1)
    num_train = len(lines) - num_val

    # 建立AlexNet模型
    model = AlexNet()

    # 保存的方式，3代保存一次
    '''
    keras.callbacks.ModelCheckpoint(filepath,monitor='val_loss',verbose=0,save_best_only=False, save_weights_only=False, mode='auto', period=1) 
    参数说明：
    filepath：字符串，保存模型的路径
    monitor：需要监视的值
    verbose：信息展示模式，0或1(checkpoint的保存信息，类似Epoch 00001: saving model to ...)
    save_best_only：当设置为True时，监测值有改进时才会保存当前的模型（ the latest best model according to the quantity monitored will not be overwritten）
    mode：‘auto’，‘min’，‘max’之一，在save_best_only=True时决定性能最佳模型的评判准则，例如，当监测值为val_acc时，模式应为max，当监测值为val_loss时，模式应为min。在auto模式下，评价准则由被监测值的名字自动推断。
    save_weights_only：若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
    period：CheckPoint之间的间隔的epoch数
    '''
    checkpoint_period1 = ModelCheckpoint(
        log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='acc',
        save_weights_only=False,
        save_best_only=True,
        period=3
    )


    # 学习率下降的方式，acc三次不下降就下降学习率继续训练
    '''
    monitor：监测的值，可以是accuracy，val_loss,val_accuracy
    factor：缩放学习率的值，学习率将以lr = lr*factor的形式被减少
    patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
    mode：‘auto’，‘min’，‘max’之一 默认‘auto’就行
    epsilon：阈值，用来确定是否进入检测值的“平原区”
    cooldown：学习率减少后，会经过cooldown个epoch才重新进行正常操作
    min_lr：学习率最小值，能缩小到的下限 
    verbose：日志冗长度，int：冗长度，0：不输出训练过程，1：偶尔输出，>1：对每个子模型都输出。
    
    Reduce=ReduceLROnPlateau(monitor='val_accuracy',
                         factor=0.1,
                         patience=2,
                         verbose=1,
                         mode='auto',
                         epsilon=0.0001,
                         cooldown=0,
                         min_lr=0)
    '''
    reduce_lr = ReduceLROnPlateau(
        monitor='acc',
        factor=0.5,
        patience=3,
        verbose=1
    )


    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    '''
    monitor: 监控指标，如val_loss
    min_delta: 认为监控指标有提升的最小提升值。如果变化值小于该值，则认为监控指标没有提升。
    patience: 在监控指标没有提升的情况下，epochs 等待轮数。等待大于该值监控指标始终没有提升，则提前停止训练。
    verbose: log输出方式
    mode: 三选一 {“auto”, “min”, “max”}，默认auto。min 模式是在监控指标值不再下降时停止训练；max 模式是指在监控指标值不再上升时停止训练；max 模式是指根据 monitor来自动选择。
    baseline: 监控指标需要到达的baseline值。如果监控指标没有到达该值，则提前停止。
    restore_best_weights: 是否加载训练过程中保存的最优模型权重，如果为False，则使用在训练的最后一步获得的模型权重。
    '''
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=1
    )

    # 交叉熵
    '''
    def compile(optimizer,
            loss=None,
             metrics=None,
             loss_weights=None,
             sample_weight_mode=None,
             weighted_metrics=None,
             target_tensors=None,
              **kwargs):
    optimizer的选项：
        SGD
        RMSprop
        Adam
        Adadelta
        Adagrad
        Adamax
        Nadam
        Ftrl
    optimizer：参数
        lr：大或等于0的浮点数，学习率        
        momentum：大或等于0的浮点数，动量参数
        decay：大或等于0的浮点数，每次更新后的学习率衰减值
        nesterov：布尔值，确定是否使用Nesterov动量
    
    metrics: 在训练和测试期间的模型评估标准。 通常你会使用 metrics = ['accuracy']。 
            要为多输出模型的不同输出指定不同的评估标准， 还可以传递一个字典，如 metrics = {'output_a'：'accuracy'}。
    loss:
        mean_squared_error或mse
        mean_absolute_error或mae
        mean_absolute_percentage_error或mape        
        mean_squared_logarithmic_error或msle        
        squared_hinge        
        hinge        
        binary_crossentropy（亦称作对数损失，logloss）
        categorical_crossentropy：亦称作多类的对数损失，注意使用该目标函数时，
                    需要将标签转化为形如(nb_samples, nb_classes)的二值序列
        sparse_categorical_crossentrop：如上，但接受稀疏标签。注意，使用该函数时仍然需要你的标签与输出值的维度相同，
                你可能需要在标签数据上增加一个维度：np.expand_dims(y,-1)
        kullback_leibler_divergence:从预测值概率分布Q到真值概率分布P的信息增益,用以度量两个分布的差异.
        cosine_proximity：即预测值与真实标签的余弦距离平均值的相反数
    '''
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=1e-3),
                  metrics=['accuracy'])

    # 一次的训练集大小
    batch_size = 128

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

    # 开始训练
    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
                        validation_steps=max(1, num_val // batch_size),
                        epochs=9,
                        initial_epoch=0,
                        callbacks=[checkpoint_period1, reduce_lr])
    model.save_weights(log_dir + 'last1.h5')
