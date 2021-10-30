import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']


def bounds225(res):
    return np.uint8(255 * normalization(res))


def normalization(res):
    mi, ma = np.min(res), np.max(res)
    res -= mi
    return res / (ma - mi)


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


def core_to_H(core, img_size: tuple):
    img_height, img_width = img_size[0], img_size[1]
    # 0. 生成h_p
    pad_core_x = (img_height - core.shape[0]) // 2
    pad_core_y = (img_width - core.shape[1]) // 2
    core = cv.copyMakeBorder(core, 1, 0, 1, 0, cv.BORDER_CONSTANT, value=0)
    core = cv.copyMakeBorder(core, pad_core_x, pad_core_x, pad_core_y, pad_core_y, cv.BORDER_CONSTANT, value=0)
    # 1. 核中心化
    core = np.fft.fftshift(core)
    # center = len(core) // 2
    # for x in range(core.shape[0]):
    #     for y in range(core.shape[1]):
    #         core[x, y] *= np.power(-1, x + y)
    # 2. DFT
    H = np.fft.fft2(core)
    # 3. 取虚部
    H = np.imag(H)
    # 4. H中心化
    H = np.fft.fftshift(H)
    # for x in range(H.shape[0]):
    #     for y in range(H.shape[1]):
    #         H[x, y] *= np.power(-1, x + y)

    return H


def power_spectrum(img):
    img_fft = np.fft.fft2(img)
    p = np.zeros(img.shape)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            p[x, y] = np.power(np.real(img_fft[x, y]), 2) + np.power(np.imag(img_fft[x, y]), 2)
    return p


def ideal_low_pass_filter(size: tuple, R):
    res = np.zeros(size)
    M, N = size[0], size[1]
    center_x = M // 2
    center_y = N // 2
    for x in range(M):
        for y in range(N):
            if np.sqrt(np.power(x-center_x, 2) + np.power(y-center_y, 2)) <= R:
                res[x, y] = 1

    return res


def gauss_low_pass_filter(size: tuple, D0):
    res = np.zeros(size)
    M, N = size[0], size[1]
    center_x = M // 2
    center_y = N // 2
    for x in range(M):
        for y in range(N):
            res[x, y] = np.exp(-1 * (np.power(x-center_x, 2) + np.power(y-center_y, 2))
                               / (2 * np.power(D0, 2)))

    return res


def butterworth_low_pass_filter(size: tuple, D0, n):
    res = np.zeros(size)
    M, N = size[0], size[1]
    center_x = M // 2
    center_y = N // 2
    for x in range(M):
        for y in range(N):
            res[x, y] = 1 / (1 + np.power(
                np.sqrt(np.power(x-center_x, 2) +
                        np.power(y-center_y, 2)) / D0, 2*n))

    return res


def ideal_high_pass_filter(size: tuple, R):
    return 1 - ideal_low_pass_filter(size, R)


def gauss_high_pass_filter(size: tuple, D0):
    return 1 - gauss_low_pass_filter(size, D0)


def butterworth_high_pass_filter(size: tuple, D0, n):
    return 1 - butterworth_low_pass_filter(size, D0, n)


def laplace_filter(size: tuple):
    res = np.zeros(size)
    M, N = size[0], size[1]
    center_x = M // 2
    center_y = N // 2
    for x in range(M):
        for y in range(N):
            res[x, y] = 1 + 4 * np.power(np.pi, 2) * (np.power(x-center_x, 2) +
                        np.power(y-center_y, 2))

    return res


def hist_equalization(img):
    pr = np.zeros(shape=256)
    height, width = img.shape
    for i in range(height):
        for j in range(width):
            pr[img[i, j]] += 1

    pr = pr / (height*width)
    sk = [round(sum(pr[:i+1]) * 255) for i in range(256)]

    res = np.zeros((height, width), np.uint8)
    for i in range(height):
        for j in range(width):
            res[i, j] = sk[img[i, j]]

    return res


def homomorphism(size: tuple, gamaH, gamaL, c, D0):
    assert gamaL < 1
    assert gamaH >= 1
    res = np.zeros(size)
    M, N = size[0], size[1]
    center_x = M // 2
    center_y = N // 2
    for x in range(M):
        for y in range(N):
            dis2 = np.power(x-center_x, 2) + np.power(y-center_y, 2)
            res[x, y] = (gamaH - gamaL) * (1 - np.exp(-1 * c * dis2 / np.power(D0, 2))) + gamaL

    return res


def gauss_band_stop_filter(size: tuple, C0, W):
    res = np.zeros(size)
    M, N = size[0], size[1]
    center_x = M // 2
    center_y = N // 2
    for x in range(M):
        for y in range(N):
            dis2 = np.power(x-center_x, 2) + np.power(y-center_y, 2)
            res[x, y] = 1 - np.exp(-1 *
                                   np.power(
                                       (dis2-np.power(C0, 2))
                                       / (np.sqrt(dis2) * W),
                                   2))

    return res


def gauss_band_pass_filter(size: tuple, C0, W):
    return 1 - gauss_band_stop_filter(size, C0, W)


def ideal_band_stop_filter(size: tuple, C0, W):
    res = np.zeros(size)
    M, N = size[0], size[1]
    center_x = M // 2
    center_y = N // 2
    for x in range(M):
        for y in range(N):
            if C0 - W / 2 <= np.sqrt(np.power(x-center_x, 2) + np.power(y-center_y, 2)) <= C0 + W / 2:
                pass
            else:
                res[x, y] = 1

    return res


def ideal_band_pass_filter(size: tuple, C0, W):
    return 1 - ideal_band_stop_filter(size, C0, W)


def butterworth_band_stop_filter(size: tuple, C0, W, n):
    res = np.zeros(size)
    M, N = size[0], size[1]
    center_x = M // 2
    center_y = N // 2
    for x in range(M):
        for y in range(N):
            dis2 = np.power(x-center_x, 2) + np.power(y-center_y, 2)
            res[x, y] = 1 / (1 + np.power(np.sqrt(dis2) * W / (dis2 - np.power(C0, 2)), 2*n))

    return res


def butterworth_band_pass_filter(size: tuple, C0, W, n):
    return 1 - butterworth_band_stop_filter(size, C0, W, n)


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


if __name__ == '__main__':
    img = cv.imread("/Users/xxh/projects/python/ml/4/car-moire-pattern.tif", 0)
    cv.imshow("img", img)
    # plt.subplot(2, 2, 1)
    # plt.imshow(img, 'gray')
    # plt.subplot(2, 2, 2)
    # plt.imshow(spectrum_conv(img, ideal_low_pass_filter(img.shape, 30)), 'gray')
    # plt.subplot(2, 2, 3)
    # plt.imshow(spectrum_conv(img, ideal_low_pass_filter(img.shape, 60)), 'gray')
    # plt.subplot(2, 2, 4)
    # plt.imshow(spectrum_conv(img, ideal_low_pass_filter(img.shape, 160)), 'gray')
    # plt.show()
    # cv.imshow("low conv1", spectrum_conv(img, ideal_low_pass_filter(img.shape, 10)).astype(np.uint8))
    # cv.imshow("low conv2", spectrum_conv(img, ideal_low_pass_filter(img.shape, 30)).astype(np.uint8))
    # cv.imshow("low conv3", spectrum_conv(img, ideal_low_pass_filter(img.shape, 60)).astype(np.uint8))
    # cv.imshow("low conv4", spectrum_conv(img, ideal_low_pass_filter(img.shape, 160)).astype(np.uint8))
    # cv.imshow("low conv5", spectrum_conv(img, ideal_low_pass_filter(img.shape, 460)).astype(np.uint8))

    # plt.subplot(2, 2, 1)
    # plt.imshow(img, 'gray')
    # plt.subplot(2, 2, 2)
    # plt.imshow(spectrum_conv(img, gauss_low_pass_filter(img.shape, 30)).astype(np.uint8), 'gray')
    # plt.subplot(2, 2, 3)
    # plt.imshow(spectrum_conv(img, gauss_low_pass_filter(img.shape, 60)).astype(np.uint8), 'gray')
    # plt.subplot(2, 2, 4)
    # plt.imshow(spectrum_conv(img, gauss_low_pass_filter(img.shape, 160)).astype(np.uint8), 'gray')
    # plt.show()
    # cv.imshow("gauss filter1", spectrum_conv(img, gauss_low_pass_filter(img.shape, 10)).astype(np.uint8))
    # cv.imshow("gauss filter2", spectrum_conv(img, gauss_low_pass_filter(img.shape, 30)).astype(np.uint8))
    # cv.imshow("gauss filter3", spectrum_conv(img, gauss_low_pass_filter(img.shape, 60)).astype(np.uint8))
    # cv.imshow("gauss filter4", spectrum_conv(img, gauss_low_pass_filter(img.shape, 160)).astype(np.uint8))

    # plt.subplot(2, 2, 1)
    # plt.imshow(img, 'gray')
    # plt.subplot(2, 2, 2)
    # plt.imshow(spectrum_conv(img, butterworth_low_pass_filter(img.shape, 30, 20)).astype(np.uint8), 'gray')
    # plt.subplot(2, 2, 3)
    # plt.imshow(spectrum_conv(img, butterworth_low_pass_filter(img.shape, 60, 20)).astype(np.uint8), 'gray')
    # plt.subplot(2, 2, 4)
    # plt.imshow(spectrum_conv(img, butterworth_low_pass_filter(img.shape, 160, 20)).astype(np.uint8), 'gray')
    # plt.show()
    # cv.imshow("butterworth1", spectrum_conv(img, butterworth_low_pass_filter(img.shape, 10*5, 2.25)).astype(np.uint8))
    # cv.imshow("butterworth2", spectrum_conv(img, butterworth_low_pass_filter(img.shape, 30*5, 2.25)).astype(np.uint8))
    # cv.imshow("butterworth3", spectrum_conv(img, butterworth_low_pass_filter(img.shape, 60*5, 2.25)).astype(np.uint8))
    # cv.imshow("butterworth4", spectrum_conv(img, butterworth_low_pass_filter(img.shape, 160*5, 2.25)).astype(np.uint8))

    # D0 = 50
    # plt.subplot(2, 2, 1)
    # plt.imshow(img, 'gray')
    # plt.subplot(2, 2, 2)
    # plt.imshow(spectrum_conv(img, ideal_high_pass_filter(img.shape, D0)).astype(np.uint8), 'gray')
    # plt.subplot(2, 2, 3)
    # plt.imshow(spectrum_conv(img, gauss_high_pass_filter(img.shape, D0)).astype(np.uint8), 'gray')
    # plt.subplot(2, 2, 4)
    # plt.imshow(spectrum_conv(img, butterworth_high_pass_filter(img.shape, D0, 4)).astype(np.uint8), 'gray')
    # plt.show()
    # cv.imshow("ideal high", spectrum_conv(img, ideal_high_pass_filter(img.shape, D0)).astype(np.uint8))
    # cv.imshow("gauss high", spectrum_conv(img, gauss_high_pass_filter(img.shape, D0)).astype(np.uint8))
    # cv.imshow("butter high", spectrum_conv(img, butterworth_high_pass_filter(img.shape, D0, 4)).astype(np.uint8))

    # cv.imshow("res.png", spectrum_conv(img, laplace_filter(img.shape)))
    # cv.imshow("gauss x", hist_equalization(spectrum_conv(img, 0.5 + 0.75 * gauss_high_pass_filter(img.shape, 70)).astype(np.uint8)))
    # cv.imshow("homomorphism", spectrum_conv(img, homomorphism(img.shape, 3, 0.4, 5, 20)))
    # cv.imshow("gauss band stop filter", gauss_band_stop_filter((1000, 1000), 100, 100))
    # cv.imshow("ideal band stop filter", ideal_band_stop_filter((1000, 1000), 100, 100))
    # cv.imshow("butter band stop filter", butterworth_band_stop_filter((1000, 1000), 100, 100, 2))
    # cv.imshow("band stop", spectrum_conv(img, band_stop_filter(img.shape, 30, 30)))
    # cv.imshow("band pass", spectrum_conv(img, band_pass_filter(img.shape, 30, 100)))
    BNRF_1 = butterworth_notch_resistant_filter(img, radius=9, uk=60, vk=80, n=4)
    BNRF_2 = butterworth_notch_resistant_filter(img, radius=9, uk=-60, vk=80, n=4)
    BNRF_3 = butterworth_notch_resistant_filter(img, radius=9, uk=60, vk=160, n=4)
    BNRF_4 = butterworth_notch_resistant_filter(img, radius=9, uk=-60, vk=160, n=4)
    BNRF = BNRF_1 * BNRF_2 * BNRF_3 * BNRF_4
    cv.imshow("Trap wave", spectrum_conv(img, BNRF))
    print("success")
    cv.waitKey(0)
    cv.destroyAllWindows()

