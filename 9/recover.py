import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage.morphology import erosion, dilation, opening, reconstruction, grey, disk


def to_255(res):
    return (res * 255).astype(np.uint8)


def test_dilation_recover():
    maker = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    mask = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    core = np.ones([3, 3])

    row, col = 2, 2
    plt.subplot(row, col, 1)
    plt.imshow(to_255(maker))
    plt.subplot(row, col, 2)
    plt.imshow(to_255(mask))
    plt.subplot(row, col, 3)
    erosioned = erosion(maker, core)
    plt.imshow(to_255((erosion(maker, core))))
    plt.subplot(row, col, 4)
    plt.imshow(to_255(cv.bitwise_or(erosioned, mask)))
    plt.show()


def recover_open():
    img = cv.imread("/Users/xxh/projects/python/ml/9/text.tif", 0)
    _, img = cv.threshold(img, 50, 255, cv.THRESH_BINARY)
    img = img / 255
    row, col = 2, 2
    plt.figure(figsize=[10, 5])
    plt.subplot(row, col, 1)
    plt.imshow(img, 'gray')
    plt.subplot(row, col, 2)
    erosioned = erosion(img, np.ones([51, 1]))
    plt.imshow(erosioned, 'gray')
    plt.subplot(row, col, 3)
    plt.imshow(opening(img, np.ones([51, 1])), 'gray')
    plt.subplot(row, col, 4)
    # 腐蚀的结果erosioned作为膨胀重建的标记
    plt.imshow(reconstruction(erosioned, img), 'gray')
    plt.show()


def gray_recover():
    img = cv.imread("/Users/xxh/projects/python/ml/9/calculator.tif", 0)
    row, col = 3, 3
    plt.figure(figsize=[20, 15])
    plt.subplot(row, col, 1)
    plt.imshow(img, 'gray')
    erosioned = grey.erosion(img, np.ones([1, 71]))
    plt.subplot(row, col, 2)
    # 重建开运算
    reconstruct_open = reconstruction(erosioned, img)
    plt.imshow(reconstruct_open, 'gray')
    plt.subplot(row, col, 3)
    # 开运算
    gray_open = grey.opening(img, np.ones([1, 71]))
    plt.imshow(gray_open, 'gray')
    plt.subplot(row, col, 4)
    # 重建顶帽运算
    reconstruct_top_hat = img - reconstruct_open
    plt.imshow(reconstruct_top_hat, 'gray')
    plt.subplot(row, col, 5)
    # 顶帽运算
    plt.imshow(img - gray_open, 'gray')
    plt.subplot(row, col, 6)
    # 重建开运算删除垂直反射，SIN中的I也被细化没了
    remove_vertical_reflect = reconstruction(grey.erosion(reconstruct_top_hat, np.ones([1, 11])), reconstruct_top_hat)
    plt.imshow(remove_vertical_reflect, 'gray')
    plt.subplot(row, col, 7)
    dilationed = grey.dilation(remove_vertical_reflect, np.ones([1, 21]))
    plt.imshow(dilationed, 'gray')
    plt.subplot(row, col, 8)
    minied = np.minimum(reconstruct_top_hat, dilationed)
    plt.imshow(minied, 'gray')
    plt.subplot(row, col, 9)
    plt.imshow(reconstruction(minied, reconstruct_top_hat), 'gray')
    plt.show()


if __name__ == '__main__':
    # test_dilation_recover()
    # recover_open()
    gray_recover()
