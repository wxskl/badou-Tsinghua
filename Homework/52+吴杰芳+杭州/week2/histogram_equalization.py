# encoding=gbk
import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
calcHist������ͼ��ֱ��ͼ
����ԭ�ͣ�calcHist(images, channels, mask, histSize, ranges, hist=None, accumulate=None)
images��ͼ��������磺[image]
channels��ͨ���������磺0
mask����Ĥ��һ��Ϊ��None
histSize��ֱ��ͼ��С��һ����ڻҶȼ���
ranges�����᷶Χ
'''
# ��ȡ�Ҷ�ͼ��
img = cv2.imread('lenna.png', 1)
# cv2.imshow("src img",img)
# cv2.waitKey()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("gray img",gray)
# cv2.waitKey()

# ԭ�Ҷ�ͼ��ֱ��ͼ
plt.figure()
plt.subplot(211)
plt.hist(gray.ravel(), 256) # ravel()����ά����ת��Ϊһά����
plt.title("ԭ�Ҷ�ͼ��ֱ��ͼ")
# �Ҷ�ͼֱ��ͼ���⻯
dst = cv2.equalizeHist(gray)

# ֱ��ͼ
hist = cv2.calcHist([dst], [0], None, [256], [0,256])
plt.subplot(212)
plt.hist(dst.ravel(), 256)
plt.title("ֱ��ͼ���⻯��ֱ��ͼ")
plt.show()

'''
np.vstack():����ֱ�����϶ѵ�
np.hstack():��ˮƽ������ƽ��
'''
# cv2.imshow("histogram equalization", np.hstack([gray, dst]))
# cv2.waitKey()

# ��ɫͼ��ֱ��ͼ���⻯
(b, g, r) = cv2.split(img)
b_hist = cv2.equalizeHist(b)
g_hist = cv2.equalizeHist(g)
r_hist = cv2.equalizeHist(r)
# �ϲ�ÿһ��ͨ��
dst_img = cv2.merge((b_hist,g_hist, r_hist))
cv2.namedWindow("src_rgb")
cv2.namedWindow("dst_rgb")
cv2.imshow("src_rgb", img)
cv2.imshow("dst_rgb", dst_img)
cv2.waitKey()
cv2.destroyAllWindows()