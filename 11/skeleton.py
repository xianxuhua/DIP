from skimage.morphology import skeletonize, thin
from scipy.ndimage.morphology import binary_hit_or_miss
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def skeleton():
    img = cv.imread("/Users/xxh/projects/python/ml/11/blood-vessels.tif", 0)
    _, img = cv.threshold(img, 0, 255, cv.THRESH_OTSU)
    img = img / 255
    row, col = 1, 3
    plt.figure(figsize=[15, 10])
    plt.subplot(row, col, 1)
    plt.imshow(img, 'gray')

    plt.subplot(row, col, 2)
    plt.title("thinning")
    plt.imshow(img + 1 - thin(img), 'gray')

    plt.subplot(row, col, 3)
    plt.title("skeletonize")
    plt.imshow(img + 1 - skeletonize(img), 'gray')

    plt.show()


if __name__ == '__main__':
    skeleton()
