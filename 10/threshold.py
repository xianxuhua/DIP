import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage.morphology import erosion


def bounds225(res):
    return np.uint8(255 * normalization(res))


def normalization(res):
    mi, ma = np.min(res), np.max(res)
    res -= mi
    return res / (ma - mi)


def relevance(matrix, core, fill_mode):
    assert len(core) % 2 == 1, 'core大小必须为奇数'
    m, n = core.shape
    height, width = matrix.shape
    res = np.zeros((height + m - 1, width + n - 1))
    m_pad = (m - 1) // 2
    n_pad = (n - 1) // 2

    pad_matrix = cv.copyMakeBorder(matrix, m_pad, m_pad, n_pad, n_pad, fill_mode)
    new_height, new_width = pad_matrix.shape

    for i in range(m_pad, new_height - m_pad):
        for j in range(n_pad, new_width - n_pad):
            res[i, j] = np.sum(core * pad_matrix[i - m_pad:i + m_pad + 1, j - n_pad:j + n_pad + 1])

    return res


def convolution(matrix, core, fill_mode=cv.BORDER_DEFAULT):
    return relevance(matrix, np.fliplr(np.flipud(core)), fill_mode)


def gauss_noise(img, mu, sigma):
    noise = np.random.normal(mu, sigma, img.shape)
    res = img + noise
    return np.uint8(bounds225(res))


def otsu_threshold():
    img = cv.imread("/Users/xxh/projects/python/ml/10/septagon.tif", 0)
    cv.imshow("img", img)
    img = gauss_noise(img, 0, 50)
    cv.imshow("noised", img)

    plt.hist(img.flatten(), bins=255, range=[0, 255], density=True)

    img = cv.GaussianBlur(img, [3, 3], 1)
    plt.hist(img.flatten(), bins=255, range=[0, 255], density=True)
    plt.show()

    _, img = cv.threshold(img, 0, 255, cv.THRESH_OTSU)

    cv.imshow("res", img)


def edge_improve_threshold():
    # TODO: laplace
    row, col = 2, 3
    plt.figure(figsize=[20, 10])
    # img = cv.imread('/Users/xxh/projects/python/ml/10/septagon-small.tif', 0)
    img = cv.imread("/Users/xxh/projects/python/ml/10/yeast-cells.tif", 0)
    plt.subplot(row, col, 1)
    plt.imshow(img, 'gray')

    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    plt.subplot(row, col, 2)
    plt.plot(hist)

    gx = cv.Sobel(np.double(img), -1, 1, 0)
    gy = cv.Sobel(np.double(img), -1, 0, 1)
    grad = np.sqrt(gx ** 2 + gy ** 2).astype(np.uint8)
    plt.subplot(row, col, 3)
    plt.imshow(grad, 'gray')

    T = np.percentile(grad, 99.5)
    _, mask = cv.threshold(grad, T, 255, cv.THRESH_BINARY)
    plt.subplot(row, col, 4)
    plt.imshow(mask, 'gray')

    # mask：需要处理的为1，不需要处理的为0
    hist = cv.calcHist([img], [0], mask, [256], [0, 256])
    plt.subplot(row, col, 5)
    plt.plot(hist)

    _, res = cv.threshold(img, 125, 255, cv.THRESH_BINARY)
    plt.subplot(row, col, 6)
    plt.imshow(res, 'gray')

    plt.show()


def multi_threshold():
    row, col = 2, 2
    img = cv.imread("/Users/xxh/projects/python/ml/10/iceberg.tif")
    plt.subplot(row, col, 1)
    plt.imshow(img, 'gray')

    plt.subplot(row, col, 2)
    plt.hist(img.flatten(), bins=255, range=[0, 255], density=True)

    # threshold: 80, 180
    img = np.where(img > 180, 255, img)
    img = np.where(img < 80, 0, img)
    img = np.where((80 <= img) & (img <= 180), 128, img)
    plt.subplot(row, col, 3)
    plt.imshow(img, 'gray')

    plt.subplot(row, col, 4)
    plt.hist(img.flatten(), bins=255, range=[0, 255], density=True)
    plt.show()


def variable_threshold():
    row, col = 1, 3
    plt.figure(figsize=[20, 10])

    # img = cv.imread("/Users/xxh/projects/python/ml/10/text-spotshade.tif", 0)
    # img = cv.imread("/Users/xxh/projects/python/ml/10/jxrg.jpeg", 0)
    img = cv.imread("/Users/xxh/projects/python/ml/10/checkerboard1024-shaded.tif", 0)

    plt.subplot(row, col, 1)
    plt.imshow(img, 'gray')

    plt.subplot(row, col, 2)
    _, otsu = cv.threshold(img, 0, 255, cv.THRESH_OTSU)
    plt.imshow(otsu, 'gray')

    plt.subplot(row, col, 3)
    adaptiveThreshold = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 25, 3)
    plt.imshow(adaptiveThreshold, 'gray')

    plt.show()


if __name__ == '__main__':
    # otsu_threshold()
    # edge_improve_threshold()
    # multi_threshold()
    # variable_threshold()
    cv.waitKey(0)
    cv.destroyAllWindows()
