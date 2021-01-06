'''
将RGB值转化为[0,1]浮点数
• 二值化：
# if (img_gray[i, j] <= 0.5):
# img_gray[i, j] = 0
# else:
# img_gray[i, j] = 1
'''


import cv2
# 以灰度图的方式读取图片
img_gray = cv2.imread("lenna.png", 0)
cv2.imshow("Scr", img_gray)
cv2.waitKey(0)

# 归一化(0,1)
img_gray = img_gray / 255
cv2.imshow("Normalization", img_gray)
cv2.waitKey(0)

# 二值化
img_gray_h,img_gray_w = img_gray.shape
for i in range(img_gray_h):
    for j in range(img_gray_w):
        if (img_gray[i][j] <= 0.5):
            img_gray[i][j] = 0
        else:
            img_gray[i][j] = 1

cv2.imshow("Binarization", img_gray)
cv2.waitKey(0)
