import random
import cv2 as cv
import numpy as np


def noise(img):
    img = img.astype(np.uint8)
    height, width, mode = img.shape
    for i in range(height):
        for j in range(width):
            for k in range(mode):
                img[i, j, k] += random.gauss(0, 1)

    return img


def avg_remove_noise(img, count):
    tar = np.zeros_like(img).astype(np.float32)
    for _ in range(count):
        tar += np.float32(noise(img))
    tar /= count
    tar = np.clip(tar, 0, 255).astype(np.uint8)
    return tar


if __name__ == '__main__':
    img = cv.imread("./shop_qrcode.jpg")
    # cv.imshow("avg_ed 2", avg_remove_noise(img, 2))
    cv.imshow("avg_ed 50", avg_remove_noise(img, 50))
    # cv.imshow("avg_ed 100", avg_remove_noise(img, 100))
    cv.imshow("noise_img", noise(img))
    # diff(noise(img), avg_remove_noise(img, 10))
    print("success")
    cv.waitKey(0)
    cv.destroyAllWindows()
