import cv2
import numpy as np

def show_image(img,str):
    #显示图片
    """
    :param img: 要显示的图片
    :param str: 显示窗口名字
    :return:
    """
    cv2.imshow(str, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def cast(number):
    #实现获取结果图坐标映射回原图的坐标的函数
    """
    :param number: number
    :return: 取整后的数
    """
    if number - int(number) < 0.5:
        number = int(number)
    else:
        number = int(number) + 1
    return number

def saturate(number, number1):
    if number >= number1:
        number = number1
    else:
        number = number

    return number


def bilinear(src, dst):
    scale_x = (src.shape[0])/(dst.shape[0]-1)
    scale_y = (src.shape[1])/(dst.shape[1]-1)
    for i in range(dst.shape[0]):
        u = (i+0.5)*(scale_x)-0.5
        m = u - int(u)
        for j in range(dst.shape[1]):
            v = (j+0.5)*(scale_y)-0.5
            n = v - int(v)
            # dst[i, j] = 127
            # print()
            dst[i, j] = (1-n)*(1-m)*src[saturate(int(u), src.shape[0]-1), saturate(int(v), src.shape[1]-1)] + \
                        (1-n)*m*src[saturate(int(u)+1, src.shape[0]-1), saturate(int(v), src.shape[1]-1)]\
                        + (1-m)*n*src[saturate(int(u), src.shape[0]-1), saturate(int(v)+1, src.shape[1]-1)] \
                        + m*n*src[saturate(int(u)+1, src.shape[0]-1), saturate(int(v)+1, src.shape[1]-1)]

    return np.array(dst, dtype=np.uint8)


def main():
    transform_height = input()
    transform_width = input()
    transform_height = int(transform_height)
    transform_width = int(transform_width)
    src = cv2.imread('lena.jpg', 0)
    # show_image(src, 'src')
    dst = np.zeros(shape=(transform_height, transform_width), dtype=np.uint8)
    print(src.shape, dst.shape)
    dst = bilinear(src, dst)
    show_image(src, 'src')
    show_image(dst, 'dst')

if __name__ == "__main__":
    main()
