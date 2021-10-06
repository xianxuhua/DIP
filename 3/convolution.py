import numpy as np
import cv2 as cv


def relevance(matrix, core, fill_mode):
    assert len(core) % 2 == 1, 'core大小必须为奇数'
    m, n = core.shape
    height, width = matrix.shape
    # res = np.zeros((height + m - 1, width + n - 1), np.uint8)
    res = np.zeros((height + m - 1, width + n - 1))
    m_pad = (m - 1) // 2
    n_pad = (n - 1) // 2

    pad_matrix = cv.copyMakeBorder(matrix, m_pad, m_pad, n_pad, n_pad, fill_mode)
    new_height, new_width = pad_matrix.shape

    for i in range(m_pad, new_height - m_pad):
        for j in range(n_pad, new_width - n_pad):
            res[i, j] = np.sum(core * pad_matrix[i-m_pad:i+m_pad+1, j-n_pad:j+n_pad+1])

    return res


def convolution(matrix, core, fill_mode=cv.BORDER_DEFAULT):
    return relevance(matrix, np.fliplr(np.flipud(core)), fill_mode)


def is_separable(core):
    E = core[0, 0]
    c = core[:, 0]
    r = core[0, :]
    v = c
    w_T = r / E
    w_T.resize(len(w_T), 1)
    return np.linalg.matrix_rank(core) == 1 and (np.outer(v, w_T) == core).all()


def gauss_core(size, K, sigma):
    core = np.zeros((size, size))
    center = size // 2
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            core[i, j] = K * np.exp(-(np.power(x, 2)+np.power(y, 2))/(2*np.power(sigma, 2)))

    core /= np.sum(core)
    return core


def laplace_sharpen(matrix, core, fill_mode=cv.BORDER_REPLICATE):
    center = len(core) // 2
    c = -1 if core[center, center] < 0 else 1
    laplace = convolution(matrix, core)
    pad_matrix = cv.copyMakeBorder(matrix, 1, 1, 1, 1, fill_mode)
    # 使用循环遍历像素自动转为uint8，因为laplace是uint8，
    # laplace * c + pad_matrix不知道是uint8，需要手动转
    return (laplace * c + pad_matrix).astype(np.uint8)


def passivation_mask(img, core, k, fill_mode=cv.BORDER_REPLICATE):
    res = convolution(img, core)
    m, n = core.shape
    m_pad = (m - 1) // 2
    n_pad = (n - 1) // 2
    img = cv.copyMakeBorder(img, m_pad, m_pad, n_pad, n_pad, fill_mode)
    diff = img - res
    return np.uint8(img + k * diff)


def gradient_sharpen(img):
    return np.sqrt(np.power(convolution(img, np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])), 2) + np.power(convolution(img, np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])), 2)).astype(np.uint8)


def box_core(size):
    return np.ones((size, size)) / size ** 2


if __name__ == '__main__':
    matrix = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    core = np.array([
        [1, 2, 3],
        [2, 4, 6],
        [3, 6, 9]
    ])
    # print(relevance(matrix, core))
    # convolution(matrix, core)
    # print(is_separable(core))
    # print(gauss_core(3, 1, 1))
    img = cv.imread("/Users/xxh/projects/python/ml/3/Fig0342(a)(contact_lens_original).tif", 0)
    cv.imshow("img", img)
    # cv.imshow("passivation1", passivation_mask(img, gauss_core(7, 1, 1), 1))
    # cv.imshow("gradient_sharpen", gradient_sharpen(img))

    # cv.imshow("passivation2", passivation_mask(img, gauss_core(7, 1, 1), 2))
    # cv.imshow("box vague", convolution(img, box_core(3)).astype(np.uint8))
    # cv.imshow("gauss vague", convolution(img, gauss_core(21, 1, 3.5)).astype(np.uint8))

    # moon_img = cv.imread("/Users/xxh/projects/python/ml/3/Fig0338(a)(blurry_moon).tif", 0)
    # cv.imshow("moon", moon_img)
    # cv.imshow("sharpening1", laplace_sharpen(moon_img, np.array([
    #                                         [0, 1, 0],
    #                                         [1, -4, 1],
    #                                         [0, 1, 0],
    #                                     ])))
    # cv.imshow("sharpening2", laplace_sharpen(moon_img, np.array([
    #                                         [1, 1, 1],
    #                                         [1, -8, 1],
    #                                         [1, 1, 1],
    #                                     ])))
    bone = cv.imread("/Users/xxh/projects/python/ml/3/bonescan.tif", 0)
    cv.imshow("res bone", convolution(gradient_sharpen(bone), box_core(5)).astype(np.uint8))
    print("success")
    cv.waitKey(0)
    cv.destroyAllWindows()
