import random
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv


def bounds225(res):
    return np.uint8(255 * normalization(res))


def normalization(res):
    mi, ma = np.min(res), np.max(res)
    res -= mi
    return res / (ma - mi)


def gauss_noise(img, mu, sigma):
    noise = np.random.normal(mu, sigma, img.shape)
    res = img + noise
    return np.uint8(bounds225(res))


def salt_pepper_noise(img, ps, pp):
    assert 0 <= ps <= 1
    assert 0 <= pp <= 1
    height, width = img.shape

    for i in range(int(height * width * ps)):
        img[random.randint(0, height-1), random.randint(0, width-1)] = 0
    for i in range(int(height * width * pp)):
        img[random.randint(0, height-1), random.randint(0, width-1)] = 255

    return img


def algorithm_operator(area):
    return np.mean(area)


def geometry_operator(area):
    area = area.astype(np.float64)
    res = np.prod(area)
    return np.power(res, 1 / (area.shape[0] * area.shape[1]))


def harmonic_operator(area):
    return area.shape[0] * area.shape[1] / np.sum(1/area)


def anti_harmonic_operator_remove_salt(area):
    Q = -1.
    res = np.sum(np.power(area, Q+1)) / np.sum(np.power(area, Q))
    return res


def anti_harmonic_operator_remove_pepper(area):
    Q = 1.
    return np.sum(np.power(area, Q+1)) / np.sum(np.power(area, Q))


def mid_operator(area):
    return np.median(area)


def max_operator(area):
    return np.max(area)


def min_operator(area):
    return np.min(area)


def mid_point_operator(area):
    return (np.max(area) + np.min(area)) / 2


def fix_alpha_operator(area):
    m, n = area.shape[0], area.shape[1]
    d = 6
    assert 0 <= d <= m * n - 1
    area = area.reshape((1, -1))[0]
    area = np.sort(area)
    area = area[d // 2:(-d+1) // 2]
    return np.sum(area) / (m * n - d)


noise_var = 0
def auto_adaption_mean_operator(area):
    center = len(area) // 2
    ratio = noise_var / np.var(area)
    ratio = 1 if ratio > 1 else ratio
    return area[center, center] - ratio * (area[center, center] - np.mean(area))


def adaptive_median_denoise(image):
    height, width = image.shape[:2]
    smax = 15
    m, n = smax, smax
    padding_h = int((m - 1) / 2)
    padding_w = int((n - 1) / 2)
    image_pad = np.pad(image, ((padding_h, m - 1 - padding_h), (padding_w, n - 1 - padding_w)), mode="edge")
    img_new = np.zeros(image.shape)

    for i in range(padding_h, height + padding_h):
        for j in range(padding_w, width + padding_w):
            sxy = 3  # 每一轮都重置
            k = int(sxy / 2)
            block = image_pad[i - k:i + k + 1, j - k:j + k + 1]
            zxy = image[i - padding_h][j - padding_w]
            zmin = np.min(block)
            zmed = np.median(block)
            zmax = np.max(block)

            if zmin < zmed < zmax:
                if zmin < zxy < zmax:
                    img_new[i - padding_h, j - padding_w] = zxy
                else:
                    img_new[i - padding_h, j - padding_w] = zmed
            else:
                while True:
                    sxy = sxy + 2
                    k = int(sxy / 2)

                    if zmin < zmed < zmax or sxy > smax:
                        break

                    block = image_pad[i - k:i + k + 1, j - k:j + k + 1]
                    zmed = np.median(block)
                    zmin = np.min(block)
                    zmax = np.max(block)

                if zmin < zmed < zmax or sxy > smax:
                    if zmin < zxy < zmax:
                        img_new[i - padding_h, j - padding_w] = zxy
                    else:
                        img_new[i - padding_h, j - padding_w] = zmed

    return np.uint8(img_new)


def avg_noise_img(img, operator):
    m, n = 3, 3
    height, width = img.shape
    res = np.zeros((height + m - 1, width + n - 1))
    m_pad = (m - 1) // 2
    n_pad = (n - 1) // 2

    pad_matrix = cv.copyMakeBorder(img, m_pad, m_pad, n_pad, n_pad, cv.BORDER_CONSTANT)
    new_height, new_width = pad_matrix.shape

    for i in range(m_pad, new_height - m_pad):
        for j in range(n_pad, new_width - n_pad):
            res[i, j] = operator(pad_matrix[i - m_pad:i + m_pad + 1, j - n_pad:j + n_pad + 1])

    return np.uint8(res)


def butterworth_notch_resistant_filter(img, uk, vk, radius=10, n=1):
    """
    create butterworth notch resistant filter, equation 4.155
    param: img:    input, source image
    param: uk:     input, int, center of the height
    param: vk:     input, int, center of the width
    param: radius: input, int, the radius of circle of the band pass filter, default is 10
    param: w:      input, int, the width of the band of the filter, default is 5
    param: n:      input, int, order of the butter worth fuction,
    return a [0, 1] value butterworth band resistant filter
    """
    M, N = img.shape[1], img.shape[0]

    u = np.arange(M)
    v = np.arange(N)
    u, v = np.meshgrid(u, v)
    DK = np.sqrt((u - M // 2 - uk) ** 2 + (v - N // 2 - vk) ** 2)
    D_K = np.sqrt((u - M // 2 + uk) ** 2 + (v - N // 2 + vk) ** 2)
    D0 = radius
    kernel = (1 / (1 + (D0 / (DK + 1e-5)) ** n)) * (1 / (1 + (D0 / (D_K + 1e-5)) ** n))

    return kernel


def spectrum_conv(matrix, filter, fill_mode=cv.BORDER_CONSTANT):
    # 1. M*N的图像，将其填充到2M*2N(直接使用空间卷积和先DFT再乘积结果相同的条件)
    pad_matrix = cv.copyMakeBorder(matrix, 0, matrix.shape[0], 0, matrix.shape[1], fill_mode)
    # 2. DFT
    pad_matrix = np.fft.fft2(pad_matrix)
    # 3. 中心化
    pad_matrix = np.fft.fftshift(pad_matrix)
    # 4. 构建大小为2M*2N的传递函数
    pad_filter_top = (pad_matrix.shape[0] - filter.shape[0] + 1) // 2
    pad_filter_bottom = (pad_matrix.shape[0] - filter.shape[0]) // 2
    pad_filter_left = (pad_matrix.shape[1] - filter.shape[1] + 1) // 2
    pad_filter_right = (pad_matrix.shape[1] - filter.shape[1]) // 2
    pad_filter = cv.copyMakeBorder(filter, pad_filter_top, pad_filter_bottom, pad_filter_left, pad_filter_right, fill_mode)
    # 5. F * H
    G = pad_matrix * pad_filter
    # 6. IDFT
    G = np.fft.ifftshift(G)
    G = np.real(np.fft.ifft2(G))
    # 7. 取大小为M*N的区域得到g(x,y)
    return bounds225(G[:matrix.shape[0], :matrix.shape[1]])


def best_trap_wave_filter(img):
    BNRF_1 = butterworth_notch_resistant_filter(img, radius=9, uk=60, vk=80, n=4)
    BNRF_2 = butterworth_notch_resistant_filter(img, radius=9, uk=-60, vk=80, n=4)
    BNRF_3 = butterworth_notch_resistant_filter(img, radius=9, uk=60, vk=160, n=4)
    BNRF_4 = butterworth_notch_resistant_filter(img, radius=9, uk=-60, vk=160, n=4)
    BNRF = BNRF_1 * BNRF_2 * BNRF_3 * BNRF_4
    eta = spectrum_conv(img, 1-BNRF)
    w = (np.mean(img * eta) - np.mean(img) * np.mean(eta)) / (np.mean(np.power(eta, 2)) - np.power(np.mean(eta), 2))
    f = img - w * eta
    cv.imshow("Trap wave", bounds225(f))


if __name__ == '__main__':
    img = cv.imread("/Users/xxh/projects/python/ml/5/img.png", 0)
    cv.imshow("img", img)
    # noise_var = 50
    # noise_img = gauss_noise(img, 0, np.sqrt(noise_var))
    # bins指定统计区间的个数, 默认只有12个
    # plt.hist(noise_img.flatten(), bins=255, range=[0, 255], density=True)

    # cv.imshow("gauss_noise", noise_img)
    # cv.imshow("algorithm_avg", avg_noise_img(noise_img, algorithm_operator))
    # cv.imshow("geometry_avg", avg_noise_img(noise_img, geometry_operator))
    # cv.imshow("harmonic_avg", avg_noise_img(noise_img, harmonic_operator))

    # salt_noise = salt_pepper_noise(img, 0.5, 0.5)
    # plt.hist(salt_noise.flatten(), bins=255, range=[0, 255], density=True)
    # plt.show()

    # cv.imshow("salt_noise", salt_noise)
    # cv.imshow("anti harmonic_avg", avg_noise_img(salt_noise, anti_harmonic_operator_remove_pepper))
    # cv.imshow("mid avg", avg_noise_img(salt_noise, mid_operator))
    # cv.imshow("max avg", avg_noise_img(salt_noise, max_operator))
    # cv.imshow("min avg", avg_noise_img(salt_noise, min_operator))
    # cv.imshow("mid point avg", avg_noise_img(noise_img, mid_point_operator))
    # cv.imshow("alpha avg", avg_noise_img(salt_noise, fix_alpha_operator))
    # img_var = np.var(img)
    # cv.imshow("auto", avg_noise_img(noise_img, auto_adaption_mean_operator))
    # cv.imshow("auto mid", adaptive_median_denoise(salt_noise))
    best_trap_wave_filter(img)
    print("success")
    cv.waitKey(0)
    cv.destroyAllWindows()
