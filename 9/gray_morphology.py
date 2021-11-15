import matplotlib.pyplot as plt
from skimage.morphology.grey import white_tophat, black_tophat, erosion, dilation, opening, closing
from skimage.morphology import square, disk
from skimage.util import random_noise
import numpy as np
import cv2 as cv


if __name__ == '__main__':
    # img = cv.imread("/Users/xxh/projects/python/ml/9/circuitboard.tif", 0)
    # cv.imshow("img", img)
    # cv.imshow("erosion", erosion(img, square(5)))
    # cv.imshow("dilation", dilation(img, square(5)))
    # # 削弱亮特征
    # cv.imshow("opening", opening(img, disk(5)))
    # # 削弱暗特征
    # cv.imshow("closing", closing(img, disk(5)))

    # img = cv.imread("/Users/xxh/projects/python/ml/9/cygnusloop.tif", 0)
    # core = disk(5)
    # cv.imshow("img", img)
    # cv.imshow("remove_noise", closing(opening(img, core), core))

    # img = cv.imread("/Users/xxh/projects/python/ml/9/headCT.tif", 0)
    # cv.imshow("img", img)
    # cv.imshow("morphological_gradient", dilation(img)-erosion(img))

    # top hat
    # img = cv.imread("/Users/xxh/projects/python/ml/9/rice-shaded.tif", 0)
    # cv.imshow("img", img)
    # _, threshold1 = cv.threshold(img, 140, 255, cv.THRESH_BINARY)
    # cv.imshow("threshold1", threshold1)
    # top_hat = img - opening(img, disk(40))
    # cv.imshow("top hat", top_hat)
    # _, threshold2 = cv.threshold(top_hat, 60, 255, cv.THRESH_BINARY)
    # cv.imshow("threshold2", threshold2)

    # 粒度测定
    # img = cv.imread("/Users/xxh/projects/python/ml/9/dowels.tif", 0)
    # cv.imshow("img", img)
    # removed_noise = closing(opening(img, disk(5)), disk(5))
    # cv.imshow("remove noise", removed_noise)
    # cv.imshow("particle", opening(removed_noise, disk(25)))

    # 纹理分割
    img = cv.imread("/Users/xxh/projects/python/ml/9/Xnip2021-11-15_15-05-44.png", 0)
    row, col = 2, 2
    plt.subplot(row, col, 1)
    plt.imshow(img, 'gray')
    # 闭运算去除小的暗区域
    closed = closing(img, disk(25))
    plt.subplot(row, col, 2)
    plt.imshow(closed, 'gray')
    # 开运算删除大的暗区域间的空隙
    opened = opening(closed, disk(60))
    plt.subplot(row, col, 3)
    plt.imshow(opened, 'gray')
    plt.subplot(row, col, 4)
    plt.imshow(img+opened, 'gray')
    plt.show()
