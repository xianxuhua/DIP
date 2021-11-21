import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from skimage.morphology import dilation


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
    cv.imshow("th", img)
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
            grayDiff = np.abs(img[currentPoint.x, currentPoint.y] - img[tmpX, tmpY])

            # 与种子是相似的
            if grayDiff < thresh and res[tmpX, tmpY] == 0:
                res[tmpX, tmpY] = label
                # 相似的作为新种子，继续生长
                seedList.append(Point(tmpX, tmpY))
    return res


def extract_connect_component_and_gen_seeds(img):
    count = 0
    core = np.ones([3, 3])
    X0 = np.zeros(img.shape, np.uint8)
    seeds = []
    while img.any():
        # 在原图上查找像素不为0的点
        xs, ys = np.where(img > 0)
        # 把第一个点赋给新图像X0
        X0[xs[0], ys[0]] = 255
        while 1:
            X1 = cv.bitwise_and(img, dilation(X0, core))
            # 找到一个连通分量
            if (X0 == X1).all():
                # 在每个连通分量上找一个像素作为种子
                xs, ys = np.where(X1 != 0)
                coordinate = list(zip(xs, ys))
                seeds.append(coordinate[len(coordinate) // 2])
                count += 1
                # 每提取到一个连通分量，原图减去该分量
                img -= X1
                break
            else:
                X0 = X1

    print("连通分量", count)
    return seeds


if __name__ == '__main__':
    row, col = 2, 3
    plt.figure(figsize=[20, 13])
    img = cv.imread("/Users/xxh/projects/python/ml/10/weldXray.tif", 0)
    plt.subplot(row, col, 1)
    plt.title("raw")
    plt.imshow(img, 'gray')

    plt.subplot(row, col, 2)
    plt.title("raw hist")
    plt.hist(img.flatten(), bins=255, range=[0, 255], density=True)

    _, th_img = cv.threshold(img, 254, 255, 0)
    plt.subplot(row, col, 3)
    plt.title("threshold")
    plt.imshow(th_img, 'gray')

    plt.subplot(row, col, 4)
    seeds = extract_connect_component_and_gen_seeds(th_img)
    seed_img = np.zeros(th_img.shape)
    for x, y in seeds:
        seed_img[x, y] = 255
    plt.title("seeds")
    plt.imshow(seed_img, 'gray')

    T = 68
    _, th1 = cv.threshold(255-img, T, 255, cv.THRESH_BINARY)
    plt.subplot(row, col, 5)
    plt.title("image inverse and threshold")
    plt.imshow(th1, 'gray')

    seeds = [Point(i[0], i[1]) for i in seeds]
    res = region_grow(th1, seeds, 1, p=0)
    plt.subplot(row, col, 6)
    plt.title("region grow")
    plt.imshow(res, 'gray')

    plt.show()
