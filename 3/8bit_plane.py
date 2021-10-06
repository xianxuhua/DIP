import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def bit_layer(img):
    h, w = img.shape
    res = np.zeros((h, w, 8))
    for i in range(h):
        for j in range(w):
            # 转为2进制
            n = str(np.binary_repr(img[i, j], 8))
            for k in range(8):
                res[i, j, k] = n[k]

    return res


def recover(img):
    b = 128
    for i in range(8):
        img[:, :, i] *= b
        b /= 2
    res = np.zeros((img.shape[0], img.shape[1]))
    for i in range(8):
        res += img[:, :, i]
    return res


if __name__ == '__main__':
    img = cv.imread("./Xnip2021-09-30_09-31-40.png", 0)
    col, row = 3, 3
    # plt.subplot(row, col, 1)
    # plt.imshow(img, cmap=plt.cm.gray)
    res = bit_layer(img)
    # for i in range(8):
    #     plt.subplot(row, col, i+2)
    #     plt.imshow(res[:, :, i], cmap=plt.cm.gray)
    print("success")
    plt.imshow(recover(res), cmap=plt.cm.gray)

    plt.show()
