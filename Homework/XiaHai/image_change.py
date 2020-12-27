import cv2
import numpy as np


def change(image: object, changeIt: object) -> object:
    # 第一个参数image为图片数据，第二个参数changeit是缩放倍数
    height, width, channels = image.shape
    if changeIt != 1:
        y_len = int(np.floor(height * changeIt))
        x_len = int(np.floor(width * changeIt))
    else:
        return image
    changeImage = np.zeros((x_len, y_len, channels), np.uint8)
    # 建立目标图像矩阵
    scale_x, scale_y = float(width) / x_len, float(height) / y_len
    # 源高宽与目标高宽比
    for i in range(channels):
        for x in range(x_len):
            for y in range(y_len):
                x0 = (x + 0.5) * scale_x - 0.5
                y0 = (y + 0.5) * scale_y - 0.5
                # 中心对齐(x0,y0)为P点坐标
                x1 = int(np.floor(x0))
                x2 = min(x1 + 1, width - 1)
                y1 = int(np.floor(y0))
                y2 = min(y1 + 1, height - 1)
                # 得出Q11,Q12,Q21,Q22坐标
                # r1 = (x2 - x0) * image[x1, y1, i] + (x0 - x1) * image[x2, y1, i]
                # r2 = (x2 - x0) * image[x1, y2, i] + (x0 - x1) * image[x2, y2, i]
                # changeImage[x, y, i] = int((y2 - y0)*r1 + (y0 - y1)*r2)
                # 带入公式，将浮点值转成整数
                changeImage[x, y, i] = int(
                    (y2 - y0) * ((x2 - x0) * image[x1, y1, i] + (x0 - x1) * image[x2, y1, i]) + (y0 - y1) * (
                                (x2 - x0) * image[x1, y2, i] + (x0 - x1) * image[x2, y2, i]))
    return changeImage


if __name__ == '__main__':
    img1 = cv2.imread("lenna.png", 1)
    changeImage = change(img1, 0.34)
    print(changeImage.shape)
    cv2.imshow("Bilinear Interpolation", changeImage)
    cv2.imshow("image", img1)
    cv2.waitKey(0)
