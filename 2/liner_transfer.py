import cv2 as cv
import numpy as np


def reverse(img):
    img = 255 - img
    return img


def gama(img, gama):
    height, width = img.shape
    res = np.zeros((height, width), np.uint8)
    for i in range(height):
        for j in range(width):
            # 先归一化，防止越界
            res[i, j] = np.power(img[i, j]/255, gama)*255
    return res


def grey_stretch(img):
    pass


if __name__ == '__main__':
    img = cv.imread("./Xnip2021-09-29_11-30-36.png", 0)
    cv.imshow("img", img)
    cv.imshow("reversed", reverse(img))
    cv.imshow("gama", gama(img, 0.2))
    print("success")
    cv.waitKey(0)
    cv.destroyAllWindows()
