import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
'''
手动还原canny边缘检测算法，根据每步步骤写出相应代码
1. 对图像进行灰度化：
    方法1： Gray=(R+G+B)/3;
    方法2： Gray=0.299R+0.587G+0.114B;（这种参数考虑到了人眼的生理特点）
2. 对图像进行高斯滤波：
    根据待滤波的像素点及其邻域点的灰度值按照一定的参数规则进行加权平均。 这样
    可以有效滤去理想图像中叠加的高频噪声。
3. 检测图像中的水平、 垂直和对角边缘（如Prewitt， Sobel算子等） 。算像素点的梯度强度
4 对梯度幅值进行非极大值抑制
5 用双阈值算法检测和连接边缘
'''


def canny(img):
    '''
    输入图像返回canny边缘检测后的图像
    :param img:
    :return:
    '''
    # 1.对图像进行灰度化处理
    img_shape = img.shape
    gray = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            gray[i, j] = 0.114*img[i, j, 0] + 0.587*img[i, j, 1] + 0.299*img[i, j, 2]  # 方法2

    # if pic_path[-4:] == '.png':  # .png图片在这里的存储格式是0到1的浮点数，所以要扩展到255再计算
    #     img = img * 255  # 还是浮点数类型
    # gray = img.mean(axis=-1)  # 取均值就是灰度化了


    # gray.astype(int)
    print(gray)
    # 2.高斯滤波,参考老师的代码
    sigma = 1.52  # 高斯平滑时的高斯核参数，标准差，可调
    dim = int(np.round(6 * sigma + 1))  # round是四舍五入函数，根据标准差求高斯核是几乘几的，也就是维度
    if dim % 2 == 0:  # 最好是奇数,不是的话加一
        dim += 1
    Gaussian_filter = np.zeros([dim, dim])  # 存储高斯核，这是数组不是列表了
    '''
    关于高斯核元素的计算：
    一个离散的高斯卷积核 H: [2k+1，2k+1]
    每个元素（i,j)的计算为： n1 * math.exp(n2 * ((i-k+1) ** 2 + (j-k+1) ** 2))
    好像(i-k)也可以
    '''
    tmp = [i - dim // 2 for i in range(dim)]  # 生成一个序列
    n1 = 1 / (2 * math.pi * sigma ** 2)  # 计算高斯核
    n2 = -1 / (2 * sigma ** 2)
    for i in range(dim):
        for j in range(dim):
            Gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
    Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()
    dx, dy = gray.shape
    img_new = np.zeros(gray.shape)  # 存储平滑之后的图像，zeros函数得到的是浮点型数据
    tmp = dim // 2
    img_pad = np.pad(gray, ((tmp, tmp), (tmp, tmp)), 'constant')  # 边缘填补
    '''
    np.pad(array, pad_width, mode, **kwargs)
        返回值：数组
        array——表示需要填充的数组；
        pad_width——表示每个轴（axis）边缘需要填充的数值数目。 
            参数输入方式为：（(before_1, after_1), … (before_N, after_N)），
            其中(before_1, after_1)表示第1轴两边缘分别填充before_1个和after_1个数值。取值为：{sequence, array_like, int}
        mode——表示填充的方式（取值：str字符串或用户提供的函数）,总共有11种填充模式；
        
        ‘constant’——表示连续填充相同的值，每个轴可以分别指定填充值，constant_values=（x, y）时前面用x填充，后面用y填充，缺省值填充0
        ‘edge’——表示用边缘值填充
        ‘linear_ramp’——表示用边缘递减的方式填充
        ‘maximum’——表示最大值填充
        ‘mean’——表示均值填充
        ‘median’——表示中位数填充
        ‘minimum’——表示最小值填充
        ‘reflect’——表示对称填充
        ‘symmetric’——表示对称填充
        ‘wrap’——表示用原数组后面的值填充前面，前面的值填充后面

    '''
    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_pad[i:i + dim, j:j + dim] * Gaussian_filter)
            pass
        pass
    plt.imsave('Gas.png', img_new)
    plt.figure(1)
    plt.imshow(img_new.astype(np.uint8), cmap='gray')  # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
    plt.axis('off')

    # 2、求梯度。以下两个是滤波求梯度用的sobel矩阵（检测图像中的水平、垂直和对角边缘）
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_tidu_x = np.zeros(img_new.shape)  # 存储梯度图像
    img_tidu_y = np.zeros([dx, dy])
    img_tidu = np.zeros(img_new.shape)
    img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')  # 边缘填补，根据上面矩阵结构所以写1
    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_x)  # x方向
            img_tidu_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)  # y方向
            img_tidu[i, j] = np.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2)
    img_tidu_x[img_tidu_x == 0] = 0.00000001
    angle = img_tidu_y / img_tidu_x
    plt.figure(2)
    plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
    plt.axis('off')

    # 3、非极大值抑制
    '''
    1) 将当前像素的梯度强度与沿正负梯度方向上的两个像素进行比较。
    2) 如果当前像素的梯度强度与另外两个像素相比最大， 则该像素点保留为边缘点， 否则
    该像素点将被抑制（灰度值置为0） 。
    '''
    img_yizhi = np.zeros(img_tidu.shape)
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            flag = True  # 在8邻域内是否要抹去做个标记
            temp = img_tidu[i - 1:i + 2, j - 1:j + 2]  # 梯度幅值的8邻域矩阵
            if angle[i, j] <= -1:  # 判断抑制与否,angle<=-1 表示该像素点的梯度强度往左上或者右下的变化最大,使用线性插值法求对应方向的两个点
                num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] >= 1:
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] < 0:
                num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            if flag:
                img_yizhi[i, j] = img_tidu[i, j]
    plt.figure(3)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')

    # 4、双阈值检测，连接边缘。遍历所有一定是边的点,查看8邻域是否存在有可能是边的点，进栈
    '''
    完成非极大值抑制后， 会得到一个二值图像， 非边缘的点灰度值均为0， 可能为边缘的局
    部灰度极大值点可设置其灰度为128。
    这样一个检测结果还是包含了很多由噪声及其他原因造成的假边缘。 因此还需要进一步的
    处理。
    如果边缘像素的梯度值高于高阈值， 则将其标记为强边缘像素；
    如果边缘像素的梯度值小于高阈值并且大于低阈值， 则将其标记为弱边缘像素；
    如果边缘像素的梯度值小于低阈值， 则会被抑制。
    '''
    lower_boundary = img_tidu.mean() * 0.5
    high_boundary = lower_boundary * 3  # 这里我设置高阈值是低阈值的三倍
    zhan = []
    for i in range(1, img_yizhi.shape[0] - 1):  # 外圈不考虑了
        for j in range(1, img_yizhi.shape[1] - 1):
            if img_yizhi[i, j] >= high_boundary:  # 取，一定是边的点
                img_yizhi[i, j] = 255
                zhan.append([i, j])
            elif img_yizhi[i, j] <= lower_boundary:  # 舍弃低于低阈值的点，即设为0
                img_yizhi[i, j] = 0

    # 抑制孤立的低阈值点
    '''
    到目前为止， 被划分为强边缘的像素点已经被确定为边缘， 因为它们是从图像中的真实边缘中提取
    出来的。
    然而， 对于弱边缘像素， 将会有一些争论， 因为这些像素可以从真实边缘提取也可以是因噪声或颜
    色变化引起的。
    为了获得准确的结果， 应该抑制由后者引起的弱边缘：
    • 通常， 由真实边缘引起的弱边缘像素将连接到强边缘像素， 而噪声响应未连接。
    • 为了跟踪边缘连接， 通过查看弱边缘像素及其8个邻域像素， 只要其中一个为强边缘像素，
      则该弱边缘点就可以保留为真实的边缘。
    '''
    while not len(zhan) == 0:
        temp_1, temp_2 = zhan.pop()  # 出栈
        a = img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
        if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2 - 1] = 255  # 这个像素点标记为边缘
            zhan.append([temp_1 - 1, temp_2 - 1])  # 进栈
        if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2] = 255
            zhan.append([temp_1 - 1, temp_2])
        if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2 + 1] = 255
            zhan.append([temp_1 - 1, temp_2 + 1])
        if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
            img_yizhi[temp_1, temp_2 - 1] = 255
            zhan.append([temp_1, temp_2 - 1])
        if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
            img_yizhi[temp_1, temp_2 + 1] = 255
            zhan.append([temp_1, temp_2 + 1])
        if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2 - 1] = 255
            zhan.append([temp_1 + 1, temp_2 - 1])
        if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2] = 255
            zhan.append([temp_1 + 1, temp_2])
        if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2 + 1] = 255
            zhan.append([temp_1 + 1, temp_2 + 1])

    for i in range(img_yizhi.shape[0]):
        for j in range(img_yizhi.shape[1]):
            if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
                img_yizhi[i, j] = 0
                pass
            pass
        pass
    plt.imsave('canny.png', img_yizhi)

img = cv2.imread('lenna.png', )

# pic_path = 'lenna.png'
# img = plt.imread(pic_path)

canny(img)