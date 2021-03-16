import cv2
import numpy as np


def drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch):
    h1, w1 = img1_gray.shape[:2]
    h2, w2 = img2_gray.shape[:2]

    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1] = img1_gray
    vis[:h2, w1:w1 + w2] = img2_gray

    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]

    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)

    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))

    cv2.namedWindow("match", cv2.WINDOW_NORMAL)
    cv2.imshow("match", vis)


img1_gray = cv2.imread("iphone1.png")
img2_gray = cv2.imread("iphone2.png")

# sift = cv2.SIFT()
sift = cv2.xfeatures2d.SIFT_create()
'''
1. sift = cv2.xfeatures2d.SIFT_create() 实例化
    参数说明：sift为实例化的sift函数
2. kp = sift.detect(gray, None)  找出图像中的关键点
    参数说明: kp表示生成的关键点，gray表示输入的灰度图，
3. ret = cv2.drawKeypoints(gray, kp, img) 在图中画出关键点
    参数说明：gray表示输入图片, kp表示关键点，img表示输出的图片
4.kp, dst = sift.compute(kp) 计算关键点对应的sift特征向量
    参数说明：kp表示输入的关键点，dst表示输出的sift特征向量，通常是128维的
'''
# sift = cv2.SURF()

kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

# BFmatcher with default parms   Brute-Force匹配器
bf = cv2.BFMatcher(cv2.NORM_L2)
# 创建BF匹配器对象 两个可选参数，第一个是normType。它指定要使用的距离量度。默认是cv2.NORM_L2欧式距离。对于SIFT,SURF很好。（还有cv2.NORM_L1）
# 当它创建以后，两个重要的方法是BFMatcher.match()和BFMatcher.knnMatch()。第一个返回最匹配的，第二个方法返回k个最匹配的，
# k由用户指定。当我们需要多个的时候很有用。
matches = bf.knnMatch(des1, des2, k=2)
'''
knnMatch返回n组两个DMatch数据结构：它包含三个非常重要的数据分别是queryIdx，trainIdx，distance
queryIdx：测试图像的特征点描述符的下标（第几个特征点描述符），同时也是描述符对应特征点的下标。
trainIdx：样本图像的特征点描述符下标,同时也是描述符对应特征点的下标。
distance：代表这怡翠匹配的特征点描述符的欧式距离，数值越小也就说明俩个特征点越相近。
这俩个DMatch数据类型是俩个与原图像特征点最接近的俩个特征点（match返回的是最匹配的）只有这俩个特征点的欧式距离小于一定值的时候才会认为匹配成功。
'''
# print(matches)

goodMatch = []
for m, n in matches:
    print(m,n)
    if m.distance < 0.50 * n.distance:
        goodMatch.append(m)

drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch[:20])

cv2.waitKey(0)
cv2.destroyAllWindows()