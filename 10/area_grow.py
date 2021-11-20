import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from skimage.morphology import erosion


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def select_connects(p):
    if p != 0:
        connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1),
                    Point(0, 1), Point(-1, 1), Point(-1, 0)]
    else:
        connects = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]
    return connects


def region_grow(img, seeds, thresh, p=1):
    height, weight = img.shape
    res = np.zeros(img.shape)
    seedList = []
    for seed in seeds:
        seedList.append(seed)
    label = 255
    connects = select_connects(p)
    while len(seedList) > 0:
        currentPoint = seedList.pop(0)

        # 种子位置为255
        res[currentPoint.x, currentPoint.y] = label
        for i in range(len(connects)):
            # 8邻域
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y
            # 若和种子不是8连通的
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                continue
            # 在原图中当前种子点和8连通的点的灰度差
            temp_point = Point(tmpX, tmpY)
            grayDiff = np.abs(img[currentPoint.x, currentPoint.y] - img[temp_point.x, temp_point.y])
            # 与种子是相似的
            if grayDiff < thresh and res[tmpX, tmpY] == 0:
                res[tmpX, tmpY] = label
                # 相似的作为新种子，继续生长
                seedList.append(Point(tmpX, tmpY))
    return res


if __name__ == '__main__':
    row, col = 3, 3
    plt.figure(figsize=[15, 10])
    img = cv.imread("/Users/xxh/projects/python/ml/10/weldXray.tif", 0)
    plt.subplot(row, col, 1)
    plt.imshow(img, 'gray')

    plt.subplot(row, col, 2)
    plt.hist(img.flatten(), bins=255, range=[0, 255], density=True)

    _, th_img = cv.threshold(img, 254, 255, 0)
    plt.subplot(row, col, 3)
    plt.imshow(th_img, 'gray')

    plt.subplot(row, col, 4)
    erosioned = erosion(th_img, np.ones([3, 3]))
    plt.imshow(erosioned, 'gray')

    plt.subplot(row, col, 5)
    diff_img = np.abs(erosioned-img)
    plt.imshow(diff_img, 'gray')

    plt.subplot(row, col, 6)
    plt.hist(diff_img.flatten(), bins=255, range=[0, 255], density=True)

    T1, T2 = 61, 126
    th2 = np.where(diff_img < T1, 0, img)
    th2 = np.where(th2 > T2, 255, img)
    th2 = np.where((th2 >= T1) & (th2 <= T2), 128, img)
    plt.subplot(row, col, 7)
    plt.imshow(th2, 'gray')

    _, th1 = cv.threshold(diff_img, T1, 255, cv.THRESH_BINARY)
    plt.subplot(row, col, 8)
    plt.imshow(th1, 'gray')

    x, y = np.where(th_img > 0)
    seeds = [Point(i[0], i[1]) for i in list(zip(x, y))]
    res = region_grow(img, seeds, 20, p=0)
    plt.subplot(row, col, 9)
    plt.imshow(res, 'gray')

    plt.show()
