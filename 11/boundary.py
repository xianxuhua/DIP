import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage.morphology.grey import erosion
from skimage.morphology import opening
from scipy.ndimage.morphology import binary_fill_holes


def find_boundary():
    img = cv.imread("/Users/xxh/projects/python/ml/11/noisy-stroke.tif", 0)
    raw = img.copy()
    row, col = 2, 3
    plt.figure(figsize=[10, 10])
    plt.subplot(row, col, 1)
    plt.imshow(img, 'gray')

    plt.subplot(row, col, 2)
    img = cv.GaussianBlur(img, [9, 9], 3)
    plt.imshow(img, 'gray')

    plt.subplot(row, col, 3)
    _, img = cv.threshold(img, 0, 255, cv.THRESH_OTSU)
    plt.imshow(img, 'gray')
    
    plt.subplot(row, col, 4)
    img = binary_fill_holes(img).astype(np.uint8)
    img = img - erosion(img, np.ones([5, 5]))
    plt.imshow(img, 'gray')

    plt.subplot(row, col, 5)
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    raw = cv.cvtColor(raw, cv.COLOR_GRAY2RGB)
    cv.drawContours(raw, contours, -1, (255, 0, 0), 3)
    plt.imshow(raw, 'gray')
    plt.show()


def approx_boundary():
    img = cv.imread("/Users/xxh/projects/python/ml/11/mapleleaf.tif")
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    row, col = 1, 2
    plt.subplot(row, col, 1)
    plt.imshow(img, 'gray')

    plt.subplot(row, col, 2)
    _, binary_img = cv.threshold(gray_img, 0, 255, cv.THRESH_OTSU)
    binary_img = opening(binary_img, np.ones([30, 30]))
    contours, hierarchy = cv.findContours(binary_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for i in contours:
        epsilon = 0.01 * cv.arcLength(i, True)
        approx = cv.approxPolyDP(i, epsilon, True)
        cv.polylines(img, [approx], True, (255, 0, 0), 2)
    plt.imshow(img, 'gray')
    
    plt.show()


if __name__ == '__main__':
    # find_boundary()
    approx_boundary()