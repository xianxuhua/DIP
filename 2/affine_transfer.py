import cv2 as cv
import numpy as np
import math


def rotate(img, angle):
    height, width, _ = img.shape
    res = np.zeros((height, width, 3), np.uint8)
    anglePi = angle * np.pi / 180.0

    for i in range(height):
        for j in range(width):
            # y = round(j * np.cos(anglePi) - i * np.sin(anglePi))
            # x = round(j * np.sin(anglePi) + i * np.cos(anglePi))
            # 后向映射，下面为逆矩阵
            srcY = (j * np.cos(anglePi) + i * np.sin(anglePi))
            srcX = (-j * np.sin(anglePi) + i * np.cos(anglePi))
            # 后向映射+双线性插值
            x = math.floor(srcX)
            y = math.floor(srcY)
            u = srcX - x
            v = srcY - y
            if 0 <= x <= 255 and 0 <= y <= 255:
                res[i, j] = (1 - u) * (1 - v) * img[x, y] + u * (1 - v) * img[x + 1, y] + (1 - u) * v * img[x, y + 1] + u * v * img[x + 1, y + 1]
                # res[i, j] = img[x, y]
                # res[x, y] = img[i, j]

    return res


if __name__ == '__main__':
    img = cv.imread("./shop_qrcode.jpg")
    cv.imshow("rotated img", rotate(img, 20))
    print("success")
    cv.waitKey(0)
    cv.destroyAllWindows()
