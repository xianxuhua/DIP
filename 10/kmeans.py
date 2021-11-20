import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import slic, mark_boundaries, find_boundaries
from skimage.morphology.grey import white_tophat, opening
from skimage.morphology.selem import square


def kmeans_cut():
    img: np.ndarray = cv.imread('/Users/xxh/projects/python/ml/10/iceberg.tif', 0)
    img = white_tophat(img, square(800))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    row, col = 2, 2
    plt.figure(figsize=[15, 10])
    plt.subplot(row, col, 1), plt.imshow(img, 'gray')

    img = slic(img, n_segments=3, start_label=1)
    plt.subplot(row, col, 2), plt.imshow(find_boundaries(img), 'gray')
    plt.subplot(row, col, 3), plt.imshow(img, 'gray')

    # 迭代参数
    # max-iter: 20, eps: 0.5
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 0.5)
    categorys = 3
    channel = 1 if len(img.shape) == 2 else img.shape[2]
    _, labels, _ = cv.kmeans(img.reshape([-1, channel]).astype(np.float32),
                             categorys, None, criteria, categorys, cv.KMEANS_RANDOM_CENTERS)
    img_output = labels.reshape((img.shape[0], img.shape[1]))
    plt.subplot(row, col, 4), plt.imshow(img_output, 'gray')
    plt.show()


if __name__ == '__main__':
    kmeans_cut()
