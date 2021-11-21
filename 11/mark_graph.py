# https://github.com/qcymkxyc/Image-Process

from scipy.spatial.distance import euclidean
import numpy as np
import cv2
from matplotlib import pyplot as plt


def get_centroid(img: np.ndarray) -> (int, int):
    """给定一张图的，返回其亮点的质心

    :param img: numpy.ndarray, 图像
    :return: (int,int), 质心的坐标
    """
    white_point = np.where([img == 1])
    x = white_point[1].mean()
    y = white_point[2].mean()

    return x, y


def get_mark_sheet(img: np.ndarray, centroid) -> np.ndarray:
    """根据图像返回标记图

    :param img: numpy.array,
    :return:
    """
    distance_list = list()

    '''获取边界'''
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contour = contours[0]
    '''距离list'''
    for point in contour:
        dist = euclidean(centroid, point)
        distance_list.append(dist)
    return distance_list


if __name__ == '__main__':
    img = cv2.imread("/Users/xxh/projects/python/ml/11/distorted-square.tif", 0)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    row, col = 1, 2
    plt.subplot(row, col, 1)
    plt.imshow(img, 'gray')

    plt.subplot(row, col, 2)
    res = get_mark_sheet(img, get_centroid(img / 255))
    plt.plot(res)

    plt.show()
