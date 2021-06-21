'''
调用cv2的api所做透视变换
'''
import cv2
import numpy as np

img = cv2.imread('photo1.jpg')

result3 = img.copy()

#img = cv2.GaussianBlur(img,(3,3),0)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 150, apertureSize = 3)
#cv2.imwrite("canny.jpg", edges)

'''
注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
'''
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
# 生成透视变换矩阵；进行透视变换
m = cv2.getPerspectiveTransform(src, dst)
result = cv2.warpPerspective(result3, m, (337, 488))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)
