import cv2 as cv
import numpy as np


def change_grey_level(img, level):
    img -= np.min(img)
    ma = np.max(img)
    height, width = img.shape
    res = np.zeros((height, width), np.uint8)
    for i in range(height):
        for j in range(width):
            res[i, j] = level * (img[i, j] / ma)

    return res


if __name__ == '__main__':
    img = cv.imread("./lic.png", 0)
    cv.imshow("img", img)
    cv.imshow("changed img", change_grey_level(img, 2))
    print("success")
    cv.waitKey(0)
    cv.destroyAllWindows()
