import cv2 as cv
import numpy as np
# import sys
# sys.path.insert(0, "/Users/xxh/projects/python/ml/6/color_transfer")


def bounds225(res):
    return np.uint8(255 * normalization(res))


def normalization(res):
    mi, ma = np.min(res), np.max(res)
    res -= mi
    return res / (ma - mi)


def color_split(rgb_img):
    r, g, b = cv.split(rgb_img)
    a = [122, 9, 24]  # 颜色拾取器获取
    height, width, mode = rgb_img.shape
    theta_r, theta_g, theta_b = np.std(r), np.std(g), np.std(b)
    res = np.zeros(rgb_img.shape, np.uint8)
    for x in range(height):
        for y in range(width):
            if a[0] - 1.25*theta_r <= rgb_img[x, y, 0] <= a[0] + 1.25*theta_r\
                and a[1] - 1.25*theta_g <= rgb_img[x, y, 1] <= a[1] + 1.25*theta_g\
                and a[2] - 1.25*theta_b <= rgb_img[x, y, 2] <= a[2] + 1.25*theta_b:
                res[x, y] = rgb_img[x, y]

    return res


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
            res[i, j] = np.sum(core * pad_matrix[i-m_pad:i+m_pad+1, j-n_pad:j+n_pad+1])

    return res


def convolution(matrix, core, fill_mode=cv.BORDER_DEFAULT):
    return relevance(matrix, np.fliplr(np.flipud(core)), fill_mode)


def colorful_convolution(rgb_img, core, fill_mode=cv.BORDER_CONSTANT):
    pad_m = core.shape[0] // 2
    pad_n = core.shape[1] // 2
    res = np.zeros(rgb_img.shape)
    res = cv.copyMakeBorder(res, pad_m, pad_m, pad_n, pad_n, fill_mode)
    res[:, :, 0] = convolution(rgb_img[:, :, 0], core, fill_mode)
    res[:, :, 1] = convolution(rgb_img[:, :, 1], core, fill_mode)
    res[:, :, 2] = convolution(rgb_img[:, :, 2], core, fill_mode)
    return res


def box_core(size):
    return np.ones((size, size)) / np.power(size, 2)


def colorful_laplace(rgb_img, core, fill_mode=cv.BORDER_CONSTANT):
    center = len(core) // 2
    c = -1 if core[center, center] < 0 else 1
    laplace = colorful_convolution(rgb_img, core, fill_mode)
    rgb_img = cv.copyMakeBorder(rgb_img, 1, 1, 1, 1, fill_mode)
    laplace[:, :, 0] = bounds225(laplace[:, :, 0] * c + rgb_img[:, :, 0])
    laplace[:, :, 1] = bounds225(laplace[:, :, 1] * c + rgb_img[:, :, 1])
    laplace[:, :, 2] = bounds225(laplace[:, :, 2] * c + rgb_img[:, :, 2])
    return laplace.astype(np.uint8)


def remove_gauss_noise(rgb_img):
    return colorful_convolution(rgb_img, np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]) / 9).astype(np.uint8)


if __name__ == '__main__':
    img = cv.imread("/Users/xxh/projects/python/ml/6/jupiter-moon-closeup.tif")
    cv.imshow("img", img)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    cv.imshow("color_split", cv.cvtColor(color_split(img), cv.COLOR_RGB2BGR))
    cv.imshow("remove noise", cv.cvtColor(remove_gauss_noise(img), cv.COLOR_RGB2BGR))

    print("success")
    cv.waitKey(0)
    cv.destroyAllWindows()
