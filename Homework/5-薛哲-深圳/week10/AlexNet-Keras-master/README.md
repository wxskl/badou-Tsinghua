# AlexNet-Keras
使用Keras实现AlexNet的模型 
 
# 使用方法Usage:
## 训练train
运行train.py即可训练。   
训练完后就会在logs里面出现模型。  
## 预测predict
将predict.py里面的模型路径改成logs里面的模型路径即可预测。  

# 自写流程：
先写模型model.py
再写一些数据处理的方法，resize之类的，utils.py
然后写处理数据集的dataset_process.py，这里的处理只是将数据集的每张图片类别统计出来，随自己
然后写trian, predict.py,后面主要是这两个写好
