import scipy.special
import numpy as np

class NeuralNetWork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        # Initialize weights
        self.wih = np.random.rand(self.hnodes,self.inodes)-0.5 #第一行表示，每个输入到第一个隐藏节点的权值
        self.who = np.random.rand(self.onodes,self.hnodes)-0.5
        # set sigmod function as activation
        self.activation_function = lambda x:scipy.special.expit(x)
        pass
    def train(self,input_list,target_list):
        inputs = np.array(input_list,ndmin=2).T
        targets = np.array(target_list,ndmin=2).T
        #信号经过输入层
        hidden_inputs = np.dot(self.wih,inputs) #inputs 要是一个二维数组，每组数表示一组输入
        #信号经过中间层
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        #Compute MSE
        # 计算误差
        output_errors = targets - final_outputs
        output_errors_1 = final_outputs*(1-final_outputs)*output_errors
        #print(len(output_errors_1),final_outputs,output_errors_1)
        hidden_errors = np.dot(self.who.T, output_errors_1)
        # 根据误差计算链路权重的更新量，然后把更新加到原来链路权重上
        self.who += self.lr * np.dot((output_errors * final_outputs * (1 - final_outputs)),
                                        np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                        np.transpose(inputs))


        pass
    def query(self,inputs):
        #print(self.wih)
        hidden_inputs = np.dot(self.wih,inputs)
        #print(hidden_inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        #print(final_outputs)
        return final_outputs
'''
data_file = open("D:/Lawrence_college/八斗/第九周/代码/NeuralNetWork_从零开始/dataset/mnist_test.csv")
data_list = data_file.readlines()
data_file.close()

all_values = data_list[0].split(',')
image_array = np.asfarray(all_values[1:]).reshape((28,28))

onodes = 10
targets = np.zeros(onodes)+0.01
targets[int(all_values[0])]=0.99
print(targets)
'''

#初始化网络
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3
n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)
#读入训练数据
#open函数里的路径根据数据存储的路径来设定
training_data_file = open("D:/Lawrence_college/八斗/第九周/代码/NeuralNetWork_从零开始/dataset/mnist_train.csv")
trainning_data_list = training_data_file.readlines()
training_data_file.close()

#设置epochs
epochs = 10
for e in range(epochs):
    for record in trainning_data_list: #
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]))/255.0 * 0.99+0.01
        targets = np.zeros(output_nodes)+0.01
        targets[int(all_values[0])]=0.99
        n.train(inputs,targets)

test_data_file = open("D:/Lawrence_college/八斗/第九周/代码/NeuralNetWork_从零开始/dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()
scores = []#define a free space
for record in test_data_list:
    all_values = record.split(",")
    correct_number = int(all_values[0])
    print("该图片对应的数字：",correct_number)
    inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 +0.01
    outputs = n.query(inputs)
    label = np.argmax(outputs)
    print("网络认为图片数字：",label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)

scores_array = np.asarray(scores) #convert a list into array
print("performance = ",scores_array.sum()/scores_array.size)





