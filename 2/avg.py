import cv2 as cv
import numpy as np


def avg(img):
    height, width = img.shape
    res = np.zeros((height, width, 3), np.uint8)
    for i in range(height):
        for j in range(width):
            if i - 1 >= 0 and j - 1 >= 0 and i + 1 <= 255 and j + 1 <= 255:
                res[i, j] = (img[i - 1, j - 1] + img[i - 1, j] + img[i - 1, j + 1] + img[i, j - 1] + img[i, j] + img[i, j + 1] + img[i + 1, j - 1] + img[i + 1, j] + img[i + 1, j + 1]) / 9

    return res


if __name__ == '__main__':
    img = cv.imread("./test.png", 0)
    cv.imshow("avg_ed img", avg(img))
    print("success")
    cv.waitKey(0)
    cv.destroyAllWindows()
