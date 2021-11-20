import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage.filters import sobel
from skimage.morphology.grey import white_tophat, black_tophat, opening


def watershed_test():
    row, col = 3, 3
    plt.figure(figsize=[20, 20])
    img = cv.imread("/Users/xxh/projects/python/ml/10/watershed_coins_01.jpg")
    plt.subplot(row, col, 1)
    plt.title("raw image")
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # # 模糊硬币上的图案
    gray = cv.GaussianBlur(gray, [15, 15], 5)
    # cv.imshow("sobel", cv.Canny(gray, 50, 150))
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # return
    plt.subplot(row, col, 2)
    plt.title("gauss")
    plt.xticks([])
    plt.yticks([])
    plt.imshow(gray, 'gray')

    plt.subplot(row, col, 3)
    _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)
    # 去除周围小的白点
    thresh = opening(thresh, np.ones([3, 3]))

    plt.title("otsu")
    plt.xticks([])
    plt.yticks([])
    plt.imshow(thresh, 'gray')

    kernel = np.ones((3, 3), np.uint8)

    # sure background area
    sure_bg = cv.dilate(thresh, kernel, iterations=2)
    plt.subplot(row, col, 4)
    plt.title("dilated")
    plt.xticks([])
    plt.yticks([])
    plt.imshow(sure_bg, 'gray')

    # 计算目标和背景的距离
    dest = cv.distanceTransform(thresh, cv.DIST_L1, 3)
    _, foreground = cv.threshold(dest, 0.5 * dest.max(), 255, cv.THRESH_BINARY)
    plt.subplot(row, col, 5)
    plt.title("dist")
    plt.xticks([])
    plt.yticks([])
    plt.imshow(dest, 'gray')

    plt.subplot(row, col, 6)
    plt.title("foreground")
    plt.xticks([])
    plt.yticks([])
    plt.imshow(foreground, 'gray')

    foreground = np.uint8(foreground)
    # 获取未知区域，栅栏会在未知区域创建
    unknown = cv.subtract(sure_bg, foreground)
    plt.subplot(row, col, 7)
    plt.title("unknown area")
    plt.xticks([])
    plt.yticks([])
    plt.imshow(unknown, 'gray')

    # Marker labelling
    _, markers = cv.connectedComponents(foreground)
    # print(markers.max())  # 连通分量
    # Add one to all labels so that sure background is not 0, but 1

    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    # 从marker开始灌水
    markers = cv.watershed(img, markers)
    plt.subplot(row, col, 8)
    plt.title("markers")
    plt.xticks([])
    plt.yticks([])
    plt.imshow(markers, 'gray')

    # 漫水算法会将找到的栅栏设置为-1
    img[markers == -1] = [255, 0, 0]
    plt.subplot(row, col, 9)
    plt.title("res")
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, 'gray')

    plt.show()


if __name__ == '__main__':
    watershed_test()
