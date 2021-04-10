import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
class Canny():
    def __init__(self,image_path):
        self.image_path = image_path

    ################################################
    #           自定义padding函数
    ################################################
    def Padding(self,image, kernels_size, stride=[1, 1], padding="same"):
        '''
        对图像进行padding
        :param image: 要padding的图像矩阵
        :param kernels_size: list 卷积核大小[h,w]
        :param stride: 卷积步长 [左右步长，上下步长]
        :param padding: padding方式
        :return: padding后的图像
        '''
        if padding == "same":
            h, w = image.shape
            p_h = max((stride[0] * (h - 1) - h + kernels_size[0]), 0)  # 高度方向要补的0
            p_w = max((stride[1] * (w - 1) - w + kernels_size[1]), 0)  # 宽度方向要补的0
            p_h_top = p_h // 2  # 上边要补的0
            p_h_bottom = p_h - p_h_top  # 下边要补的0
            p_w_left = p_w // 2  # 左边要补的0
            p_w_right = p_w - p_w_left  # 右边要补的0
            # print(p_h_top,p_h_bottom,p_w_left,p_w_right)     # 输出padding方式
            padding_image = np.zeros((h + p_h, w + p_w), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    padding_image[i + p_h_top][j + p_w_left] = image[i][j]  # 将原来的图像放入新图中做padding
            return padding_image
        else:
            return image

    #######################################################################################
    #           灰度化
    #######################################################################################
    def gray(self):
        '''
        :param img: RGB 图
        :return: 灰度图（0,255）
        对于彩色转灰度，有一个很著名的心理学公式：
        Gray = B*0.114 + G*0.587 + R*0.299
        plt函数是rgb方式读取的
        cv2函数是bgr方式读取的
        '''
        # 读取图片
        img = cv2.imread(self.image_path)
        imgInfo = img.shape
        gray = np.zeros((imgInfo[0], imgInfo[1]), dtype=np.uint8)  # gray.dtype 为 uint8  # 创建矩阵来保存变换后的图片
        gray.astype(int)
        for i in range(imgInfo[0]):
            for j in range(imgInfo[1]):
                gray[i][j] = img[i][j][0] * 0.114 + img[i][j][1] * 0.587 + img[i][j][2] * 0.299
        return gray
        # return cv2.imread(self.image_path,0)

    #######################################################################################
    #           高斯平滑滤波
    #######################################################################################
    def gaussian_smooth_filter(self,img_gray):

        # 去除噪音 - 使用 5x5 的高斯滤波器
        """
        要生成一个 (2k+1)x(2k+1) 的高斯滤波器，滤波器的各个元素计算公式如下：
        H[i, j] = (1/(2*pi*sigma**2))*exp(-1/2*sigma**2((i-k-1)**2 + (j-k-1)**2))
        """
        # 生成高斯滤波器
        sigma1 = sigma2 = 1.52   # 标准差设置
        gau_sum = 0
        dim = 5         # 高斯卷积核大小
        k = (dim-1)/2
        Gaussian_filter = np.zeros([dim, dim])
        for i in range(dim):
            for j in range(dim):
                Gaussian_filter[i, j] = math.exp((-1 / (2 * sigma1 * sigma2)) * (np.square(i - k -1)+ np.square(j - k -1))) /(2 * math.pi * sigma1 * sigma2)
                gau_sum = gau_sum + Gaussian_filter[i, j]
        # 归一化处理,获得高斯滤波器
        Gaussian_filter = Gaussian_filter / gau_sum

        # 高斯滤波
        H,W = img_gray.shape
        new_gray = np.zeros(img_gray.shape)
        img_gray =  self.Padding(img_gray,kernels_size=Gaussian_filter.shape,stride=[1,1],padding="same")
        for i in range(H):
            for j in range(W):
                new_gray[i,j] = (np.sum(img_gray[i:i+dim, j:j+ dim] * Gaussian_filter))
        # new_gray = new_gray/255
        return new_gray
    #######################################################################################
    #           Sobel算子计算梯度
    #######################################################################################
    def sobel_filter(self,image):
        h = image.shape[0]
        w = image.shape[1]
        # image.astype(np.uint8)
        sobel_filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_filter_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        image_padding = self.Padding(image,kernels_size=sobel_filter_x.shape,stride=[1,1],padding="same")
        image_gradient_value = np.zeros(image.shape)
        image_gradient_direction = np.zeros(image.shape)
        for i in range(h):
            for j in range(w):
                dx = np.sum(image_padding[i:i+3, j:j+ 3] * sobel_filter_x)
                dy = np.sum(image_padding[i:i+3, j:j+ 3] * sobel_filter_y)
                image_gradient_value[i][j] = np.sqrt(np.square(dx) + np.square(dy))
                image_gradient_direction[i][j] = dy/(dx+0.000000001)
        return image_gradient_value,image_gradient_direction
    #######################################################################################
    #          根据梯度方向角对梯度幅值进行非极大值抑制，梯度方向角image_gradient_direction
    #######################################################################################
    def Non_maximum_suppression(self,image_gradient_value,image_gradient_direction):
        # 梯度插值，计算dTmp1和dTmp2,比较梯度 并判断是否抑制
        H,W = image_gradient_value.shape
        img_NMS = np.zeros([H,W])
        for i in range(1,H-1):
            for j in range(1,W-1):
                flag = True  # 在8邻域内是否要抹去做个标记
                temp = image_gradient_value[i - 1:i + 2, j - 1:j + 2]  # 梯度幅值的8邻域矩阵
                angle = np.abs(image_gradient_direction[i, j])
                # 情况 1
                if image_gradient_direction[i, j] < -1:  # 使用线性插值法判断抑制与否
                    dTmp1 = (temp[0, 0] - temp[0, 1])/angle + temp[0, 1]
                    dTmp2 = (temp[2, 2] - temp[2, 1])/angle + temp[2, 1]
                    if not (image_gradient_value[i, j] > dTmp1 and image_gradient_value[i, j] > dTmp2):
                        flag = False
                # 情况 2
                elif image_gradient_direction[i, j] > 1:

                    dTmp1 = (temp[0, 2] - temp[0, 1])/angle + temp[0, 1]
                    dTmp2 = (temp[2, 0] - temp[2, 1])/angle + temp[2, 1]
                    if not (image_gradient_value[i, j] > dTmp1 and image_gradient_value[i, j] > dTmp2):
                        flag = False
                # 情况 3
                elif image_gradient_direction[i, j] >= 0:
                    dTmp1 = (temp[0, 2] - temp[1, 2]) * angle + temp[1, 2]
                    dTmp2 = (temp[2, 0] - temp[1, 0]) * angle + temp[1, 0]
                    if not (image_gradient_value[i, j] > dTmp1 and image_gradient_value[i, j] > dTmp2):
                        flag = False
                # 情况 4
                elif image_gradient_direction[i, j] < 0:
                    dTmp1 = (temp[0, 0] - temp[1, 0]) * angle + temp[1, 0]
                    dTmp2 = (temp[2, 2] - temp[2, 1]) * angle + temp[1, 2]
                    if not (image_gradient_value[i, j] > dTmp1 and image_gradient_value[i, j] > dTmp2):
                        flag = False
                if flag:
                    img_NMS[i, j] = image_gradient_value[i, j]
        return img_NMS


    #######################################################################################
    #          根据梯度幅值进行的非极大值抑制结果，进行双阈值算法连接边缘，遍历所有一定是边的点,查看8邻域是否存在有可能是边的点，进栈
    #######################################################################################
    def double_threshold(self, NMS, gradient):
        lower_boundary = gradient.mean() * 0.5
        high_boundary = lower_boundary * 3  # 这里我设置高阈值是低阈值的三倍
        zhan = []
        for i in range(1, NMS.shape[0] - 1):  # 外圈不考虑了
            for j in range(1, NMS.shape[1] - 1):
                if NMS[i, j] >= high_boundary:  # 取，一定是边的点，强边缘
                    NMS[i, j] = 255
                    zhan.append([i, j])
                elif NMS[i, j] <= lower_boundary:  # 舍 不是边缘
                    NMS[i, j] = 0

        while not len(zhan) == 0:
            temp_1, temp_2 = zhan.pop()  # 出栈
            a = NMS[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]   # 获得强边缘的邻域像素的梯度
            if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):  # 如果 强边缘的邻域像素img_yizhi[temp_1 - 1, temp_2 - 1]是弱边缘
                NMS[temp_1 - 1, temp_2 - 1] = 255                   # 则标记该弱边缘像素img_yizhi[temp_1 - 1, temp_2 - 1]为强边缘，并将新得到的强边缘入栈，以此类推查看强边缘点8邻域的其他像素点
                zhan.append([temp_1 - 1, temp_2 - 1])  # 进栈
            if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
                NMS[temp_1 - 1, temp_2] = 255
                zhan.append([temp_1 - 1, temp_2])
            if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
                NMS[temp_1 - 1, temp_2 + 1] = 255
                zhan.append([temp_1 - 1, temp_2 + 1])
            if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
                NMS[temp_1, temp_2 - 1] = 255
                zhan.append([temp_1, temp_2 - 1])
            if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
                NMS[temp_1, temp_2 + 1] = 255
                zhan.append([temp_1, temp_2 + 1])
            if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
                NMS[temp_1 + 1, temp_2 - 1] = 255
                zhan.append([temp_1 + 1, temp_2 - 1])
            if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
                NMS[temp_1 + 1, temp_2] = 255
                zhan.append([temp_1 + 1, temp_2])
            if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
                NMS[temp_1 + 1, temp_2 + 1] = 255
                zhan.append([temp_1 + 1, temp_2 + 1])
        # 将不在强边缘邻域内的弱边缘的像素值置0
        for i in range(NMS.shape[0]):
            for j in range(NMS.shape[1]):
                if NMS[i, j] != 0 and NMS[i, j] != 255:
                    NMS[i, j] = 0
        return NMS
    def canny(self):
        canny = Canny("lenna.png")
        img_gray = canny.gray()
        Gaussian = canny.gaussian_smooth_filter(img_gray=img_gray)
        gradient, direction = canny.sobel_filter(Gaussian)
        img_NMS = canny.Non_maximum_suppression(gradient, direction)
        threshold = canny.double_threshold(img_NMS, gradient)
        plt.figure()
        plt.axis('off')
        plt.imshow(threshold, cmap='gray')
        plt.show()


# canny = Canny("lenna.png").canny()
canny = Canny("lenna.png")
img_gray = canny.gray()
Gaussian = canny.gaussian_smooth_filter(img_gray=img_gray)
plt.figure(1)
plt.axis('off')
plt.imshow(Gaussian,cmap="gray")
gradient,direction = canny.sobel_filter(Gaussian)
plt.figure(2)
plt.axis('off')
plt.imshow(gradient,cmap="gray")

img_NMS = canny.Non_maximum_suppression(gradient, direction)

plt.figure(3)
plt.axis('off')
plt.imshow(img_NMS, cmap='gray')
threshold = canny.double_threshold(img_NMS,gradient)
plt.figure(4)
plt.axis('off')
plt.imshow(threshold, cmap='gray')

plt.show()

