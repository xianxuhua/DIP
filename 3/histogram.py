import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def hist_equalization(img):
    pr = np.zeros(shape=256)
    height, width = img.shape
    for i in range(height):
        for j in range(width):
            pr[img[i, j]] += 1

    pr = pr / (height*width)
    sk = [round(sum(pr[:i+1]) * 255) for i in range(256)]

    res = np.zeros((height, width), np.uint8)
    for i in range(height):
        for j in range(width):
            res[i, j] = sk[img[i, j]]

    return res


def sub_hist_equalization(img):
    return cv.createCLAHE(tileGridSize=(3, 3)).apply(img)


def sub_hist_equalization_with_statistic(img):
    height, width = img.shape
    k0, k1, k2, k3, C = 0, 0.1, 0, 0.1, 22.8
    m_g = np.mean(img)
    sigama_g = np.var(img)
    s_size = 3
    res = np.zeros((height, width), np.uint8)
    for i in range(height):
        for j in range(width):
            m_s = np.mean(img[i:i+s_size, j:j+s_size])
            sigama_s = np.var(img[i:i+s_size, j:j+s_size])
            if k0 * m_g <= m_s <= k1 * m_g and k2 * sigama_g <= sigama_s <= k3 * sigama_g:
                res[i, j] = C * img[i, j]
            else:
                res[i, j] = img[i, j]

    return res


if __name__ == '__main__':
    img = cv.imread("/Users/xxh/projects/python/ml/3/aerialview-washedout.tif", 0)
    # cv.imshow("raw", img)
    # cv.imshow("res2", hist_equalization(img))
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # plt.subplot(2, 1, 1)
    # plt.hist(img)
    # plt.xlim([0, 255])
    # plt.subplot(2, 1, 2)
    # plt.hist(hist_equalization(img))
    # plt.xlim([0, 255])
    hist_equalization(img)
    plt.show()
