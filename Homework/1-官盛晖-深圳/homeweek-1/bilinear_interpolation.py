"""
date: 2020.12.27
author: gsh
bilinear_interpolation
"""

"""
# linear_interpolation
# (x0,y0) (x,y) (x1,y1)
# (y-y0)/(x-x0)=(y1-y0)/(x1-x0) --> y=y0*(x1-x)/(x1-x0) + y1*(x-x0)/(x1-x0)

"""

# bilinear interpolation
#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
# 512*512 --> 700*700

def bilinear_interpolation(img, out_dim):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim[0], out_dim[1]
    print('src_h, src_w =', src_h, src_w)
    print('dst_h, dst_w =', dst_h, dst_w)
    # initialization
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.unit8)
    # for cv img, the dtype of its numpy.ndarray must be np.unit8
    # calculate the scaling
    scale_x, scale_y = float(src_w)/dst_w, float(src_h)/dst_h
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # find the corresponding src pixel
                src_x = scale_x*(dst_x + 0.5) - 0.5
                src_y = scale_y*(dst_y + 0.5) - 0.5
                # find the pixels around the src pixel
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)
                # calculate the interpolation values
                # r1 = (src_x1 - src_x) / (src_x1 - src_x0) * img[src_x0, src_y0] + (src_x - src_x0) / (src_x1 - src_x0) * img[src_x1, src_y0]
                # r2 = (src_x1 - src_x) / (src_x1 - src_x0) * img[src_x1, src_y0] + (src_x - src_x0) / (src_x1 - src_x0) * img[src_x1, src_y1]
                # img[height, width, channel] or img[y,x,n]
                r1 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                r2 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_x, dst_y] = (src_y1 - src_y) * r1 + (src_y - src_y0) * r2
    return dst_img
if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    dst = bilinear_interpolation(img, (700, 700))
    cv2.imshow('after bilinear interpolation:', dst)
    cv2.waitKey()







