# -*- encoding=UTF-8 -*-
import cv2
import numpy as np
from skimage import util

#均值hash
def avg_hash(img_input):
    img1 = cv2.resize(cv2.cvtColor(img_input,cv2.COLOR_BGR2GRAY),(8,8),interpolation=cv2.INTER_CUBIC)
    avg_num = np.mean(img1)
    #print("avg_num:",avg_num)
    img2 = np.zeros(img1.shape)
    img3 = img1.flatten()
    img2 = ['1' if i > avg_num else '0' for i in img3]
    return ''.join(img2)

#差值hash
def reduce_hash(img_input):
    img1 = cv2.resize(cv2.cvtColor(img_input,cv2.COLOR_BGR2GRAY),(9,8),interpolation=cv2.INTER_CUBIC)
    result = ''
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1] - 1):
            if img1[i,j] < img1[i,j+1]:
                result += '0'
            else:
                result += '1'
    return ''.join(result)

def cmp_diff(hash1,hash2):
    diff_count = 0
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            diff_count +=1

    return diff_count

if __name__ == "__main__":
    img_orignal = cv2.imread("images/lenna.png")
    img_noise = np.array(util.random_noise(img_orignal,mode='s&p'),np.float32)
    hash_orignal = avg_hash(img_orignal)
    hash_noise = avg_hash(img_noise)
    print("avg_hash  img_orignal:", hash_orignal)
    print("avg_hash  img_noise:", hash_noise)
    print("cmp_diff:",cmp_diff(hash_orignal,hash_noise))

    hash_orignal = reduce_hash(img_orignal)
    hash_noise = reduce_hash(img_noise)
    print("reduce_hash  img_orignal:", hash_orignal)
    print("reduce_hash  img_noise:", hash_noise)
    print("cmp_diff:",cmp_diff(hash_orignal,hash_noise))

