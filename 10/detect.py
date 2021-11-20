import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


def bounds225(res):
    return np.uint8(255 * normalization(res))


def normalization(res):
    mi, ma = np.min(res), np.max(res)
    res -= mi
    return res / (ma - mi)


def gauss_core(size, K, sigma):
    core = np.zeros((size, size))
    center = size // 2
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            core[i, j] = K * np.exp(-(np.power(x, 2) + np.power(y, 2)) / (2 * np.power(sigma, 2)))

    core /= np.sum(core)
    return core


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


def point_detect():
    img = cv.imread("/Users/xxh/projects/python/ml/10/turbineblad-with-blk-dot.tif", 0)
    laplace = convolution(img, np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1],
    ]))
    res = np.where(laplace > np.max(laplace) * 0.9, 255, 0)
    cv.imshow("point_detect", res.astype(np.uint8))


def line_detect():
    img = cv.imread("/Users/xxh/projects/python/ml/10/wirebond-mask.tif", 0)
    cv.imshow("img", img)
    core = np.array([
        [2, -1, -1],
        [-1, 2, -1],
        [-1, -1, 2],
    ])
    laplace = convolution(img, core)
    laplace = bounds225(laplace)
    cv.imshow("laplace", np.where(laplace > np.max(laplace) * 0.9, 255, 0).astype(np.uint8))


def sobel_edge_detect():
    img = cv.imread("/Users/xxh/projects/python/ml/10/building-600by600.tif", 0)
    img = cv.boxFilter(img, -1, [3, 3])
    x_edge = cv.Sobel(img, -1, 1, 0)
    y_edge = cv.Sobel(img, -1, 0, 1)
    cv.imshow("x_edge", x_edge)
    cv.imshow("y_edge", y_edge)
    cv.imshow("edge", x_edge + y_edge)


def kirsch_edge_detect():
    img = cv.imread("/Users/xxh/projects/python/ml/10/building-600by600.tif", 0)
    res1 = convolution(img, np.array([
        [5, 5, 5],
        [-3, 0, -3],
        [-3, -3, -3]
    ]))
    cv.imshow("res1", bounds225(res1))
    res2 = convolution(img, np.array([
        [5, 5, -3],
        [5, 0, -3],
        [-3, -3, -3]
    ]))
    res2 = bounds225(res2)
    cv.imshow("res2", res2)
    cv.imshow("res2 th", np.where(res2 > np.max(res2) * 0.33, 0, 255).astype(np.uint8))


def log_edge_detect():
    img = cv.imread("/Users/xxh/projects/python/ml/10/building-600by600.tif", 0)
    cv.imshow("img", img)
    img = convolution(img, gauss_core(5, 1, 4))
    laplace = convolution(img, np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1],
    ]))
    cv.imshow("laplace", laplace)
    res = np.zeros(laplace.shape)
    threshold = np.max(laplace) * 0.04

    # 寻找过零点，水平、垂直、对角线方向
    for x in range(1, laplace.shape[0] - 1):
        for y in range(1, laplace.shape[1] - 1):
            if laplace[x - 1, y] * laplace[x + 1, y] < 0 and np.abs(laplace[x - 1, y] - laplace[x + 1, y]) > threshold \
                    or laplace[x, y - 1] * laplace[x, y + 1] < 0 and np.abs(
                laplace[x, y - 1] - laplace[x, y + 1]) > threshold \
                    or laplace[x - 1, y - 1] * laplace[x + 1, y + 1] < 0 and np.abs(
                laplace[x - 1, y - 1] - laplace[x + 1, y + 1]) > threshold \
                    or laplace[x - 1, y + 1] * laplace[x + 1, y - 1] < 0 and np.abs(
                laplace[x - 1, y + 1] - laplace[x + 1, y - 1]) > threshold:
                res[x, y] = 255

    cv.imshow("res", res.astype(np.uint8))


def dog_edge_detect():
    img = cv.imread("/Users/xxh/projects/python/ml/10/building-600by600.tif", 0)
    cv.imshow("img", img)
    sigma = 2
    k = 1.6
    core_size = 5
    res = cv.GaussianBlur(img, [core_size, core_size], sigma * k) - cv.GaussianBlur(img, [core_size, core_size], sigma)
    cv.imshow("res", bounds225(res).astype(np.uint8))


def canny_edge_test():
    img = cv.imread("/Users/xxh/projects/python/ml/10/headCT.tif", 0)
    cv.imshow("img", img)
    img = convolution(img, gauss_core(13, 1, 2))
    # gx = convolution(img, np.array([
    #     [-1, -2, -1],
    #     [0, 0, 0],
    #     [1, 2, 1]
    # ]))
    # gy = convolution(img, np.array([
    #     [-1, 0, 1],
    #     [-2, 0, 2],
    #     [-1, 0, 1]
    # ]))
    # img = np.sqrt(np.power(gx, 2) + np.power(gy, 2))
    # alpha = np.arctan2(gy, gx)
    # angle = alpha * 180 / np.pi
    # res = np.zeros(angle.shape)
    #
    # # 非极大值抑制
    # l, r = 0, 0
    # for x in range(1, angle.shape[0]-1):
    #     for y in range(1, angle.shape[1]-1):
    #         current_angle = angle[x, y]
    #         if np.abs(current_angle) <= 22.5 or np.abs(current_angle) >= 157.5:
    #             # 水平边缘
    #             l, r = img[x, y-1], img[x, y+1]
    #         elif 67.5 <= np.abs(current_angle) <= 112.5:
    #             # 垂直边缘
    #             l, r = img[x-1, y], img[x+1, y]
    #         elif 112.5 <= current_angle <= 157.5 or -67.5 <= current_angle <= -22.5:
    #             # 45度
    #             l, r = img[x-1, y+1], img[x+1, y-1]
    #         elif 22.5 <= current_angle <= 67.5 or -157.5 <= current_angle <= -112.5:
    #             # -45度
    #             l, r = img[x-1, y-1], img[x+1, y+1]
    #         if img[x, y] < l or img[x, y] < r:
    #             res[x, y] = 0
    #         else:
    #             res[x, y] = img[x, y]
    #
    # # 阈值处理
    # high = 220
    # low = 100
    # weak = 1
    # res = np.where(res > high, 255, res)
    # res = np.where(res > low, weak, res)
    #
    # for x in range(1, res.shape[0]-1):
    #     for y in range(1, res.shape[1]-1):
    #         # 8邻域
    #         if res[x-1, y-1] == weak or res[x-1, y] == weak or res[x, y-1] == weak or res[x+1, y] == weak\
    #             or res[x, y+1] == weak or res[x+1, y+1] == weak or res[x-1, y+1] == weak or res[x+1, y-1] == weak:
    #             res[x, y] = 255
    #         else:
    #             res[x, y] = 0
    # cv.imshow("res", res)
    #
    cv.imshow("canny", cv.Canny(img.astype(np.uint8), 70, 200))


if __name__ == '__main__':
    # point_detect()
    # line_detect()
    # sobel_edge_detect()
    # kirsch_edge_detect()
    # log_edge_detect()
    # dog_edge_detect()
    canny_edge_test()
    cv.waitKey(0)
    cv.destroyAllWindows()
