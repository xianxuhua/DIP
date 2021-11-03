import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


def bounds225(res):
    return np.uint8(255 * normalization(res))


def normalization(res):
    mi, ma = np.min(res), np.max(res)
    res -= mi
    return res / (ma - mi)


def rgb_to_cmy(rgb_img):
    return 1 - rgb_img


def cmy_to_cmyk(cmy_img):
    cmy_img = normalization(cmy_img)
    height, width, _ = cmy_img.shape
    c, m, y = cv.split(cmy_img)
    res = np.zeros((height, width, 4), np.uint8)
    for i in range(height):
        for j in range(width):
            C, M, Y = c[i, j], m[i, j], y[i, j]
            K = min(C, M, Y)
            if K == 1:
                C, M, Y = 0, 0, 0
            else:
                C = (C - K) / (1 - K)
                M = (M - K) / (1 - K)
                Y = (Y - K) / (1 - K)
            res[i, j, 0] = C * 255
            res[i, j, 1] = M * 255
            res[i, j, 2] = Y * 255
            res[i, j, 3] = K * 255

    return res


def cmyk_to_cmy(cmyk_img):
    cmyk_img = normalization(cmyk_img)
    c, m, y, k = cv.split(cmyk_img)
    height, width, _ = cmyk_img.shape
    res = np.zeros((height, width, 3), np.uint8)
    for i in range(height):
        for j in range(width):
            C, M, Y, K = c[i, j], m[i, j], y[i, j], k[i, j]
            C = C * (1 - K) + K
            M = M * (1 - K) + K
            Y = Y * (1 - K) + K
            res[i, j, 0] = C * 255
            res[i, j, 1] = M * 255
            res[i, j, 2] = Y * 255

    return res


def rgb_to_hsi(rgb_img):
    rgb_img = normalization(rgb_img)
    r, g, b = cv.split(rgb_img)
    height, width, _ = rgb_img.shape
    res = np.zeros(rgb_img.shape)
    for i in range(height):
        for j in range(width):
            R, G, B = r[i, j], g[i, j], b[i, j]
            bt = np.sqrt(np.power(R - G, 2) + (R - B) * (G - B))
            theta = np.arccos((2 * R - G - B) / 2 / bt)
            if bt == 0:
                H = 0
            elif B <= G:
                H = theta
            else:
                H = 2 * np.pi - theta
            H /= (2 * np.pi)
            S = 1 - 3 * min(R, G, B) / (R + G + B)
            I = (R + G + B) / 3
            res[i, j, 0] = H
            res[i, j, 1] = S
            res[i, j, 2] = I

    return res


def hsi_to_rgb(hsi_img):
    h, s, i = cv.split(hsi_img)
    height, width, _ = hsi_img.shape
    res = np.zeros(hsi_img.shape, np.uint8)
    for x in range(height):
        for y in range(width):
            H, S, I = h[x, y], s[x, y], i[x, y]
            H *= (2 * np.pi)
            if 0 <= H < (2 / 3 * np.pi):
                B = I * (1 - S)
                R = I * (1 + S * np.cos(H) / np.cos(1 / 3 * np.pi - H))
                G = 3 * I - (R + B)
            elif (2 / 3 * np.pi) <= H < (4 / 3 * np.pi):
                H -= (2 / 3 * np.pi)
                R = I * (1 - S)
                G = I * (1 + S * np.cos(H) / np.cos(1 / 3 * np.pi - H))
                B = 3 * I - (R + G)
            else:
                H -= (4 / 3 * np.pi)
                G = I * (1 - S)
                B = I * (1 + S * np.cos(H) / np.cos(1 / 3 * np.pi - H))
                R = 3 * I - (G + B)

            res[x, y, 0] = R * 255
            res[x, y, 1] = G * 255
            res[x, y, 2] = B * 255

    return res


def gray_to_colorful(gray_img):
    bounds = 255
    layers = 5
    height, width = gray_img.shape
    R = np.zeros(gray_img.shape)
    G = np.zeros(gray_img.shape)
    B = np.zeros(gray_img.shape)
    for i in range(height):
        for j in range(width):
            if gray_img[i, j] < bounds / layers:
                R[i, j] = 0
                G[i, j] = 0
                B[i, j] = 0
            elif gray_img[i, j] < bounds * 2 / layers:
                R[i, j] = 0
                G[i, j] = 0
                B[i, j] = bounds
            elif gray_img[i, j] < bounds * 3 / layers:
                R[i, j] = bounds
                G[i, j] = 0
                B[i, j] = 0
            elif gray_img[i, j] < bounds * 4 / layers:
                R[i, j] = bounds
                G[i, j] = bounds
                B[i, j] = 0
            else:
                R[i, j] = bounds
                G[i, j] = bounds
                B[i, j] = bounds

    res = np.zeros((height, width, 3))
    res[:, :, 0] = R
    res[:, :, 1] = G
    res[:, :, 2] = B
    return res


def change_light(rgb_img, k):
    hsi = rgb_to_hsi(rgb_img)
    hsi[:, :, 2] *= k
    return hsi_to_rgb(hsi)


def change_saturation(rgb_img, k):
    hsi = rgb_to_hsi(rgb_img)
    hsi[:, :, 1] *= k
    return hsi_to_rgb(hsi)


def colorful_select_area(rgb_img, R):
    rgb_img = normalization(rgb_img)
    height, width, mode = rgb_img.shape
    res = np.zeros(rgb_img.shape, np.uint8)
    rgb_center = [0.6863, 0.1608, 0.1922]
    for i in range(height):
        for j in range(width):
            t = np.power(rgb_img[i, j, 0] - rgb_center[0], 2) + np.power(rgb_img[i, j, 1] - rgb_center[1], 2) + np.power(rgb_img[i, j, 2] - rgb_center[2], 2)
            if t > np.power(R, 2):
                res[i, j] = 1 * 255
            else:
                res[i, j] = rgb_img[i, j] * 255

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


def colorful_hist_equalization(rgb_img):
    img = (rgb_to_hsi(rgb_img) * 255).astype(np.uint8)
    img[:, :, 2] = hist_equalization(img[:, :, 2])
    img[:, :, 1] = img[:, :, 1] * 0.7
    return hsi_to_rgb(img / 255)



if __name__ == '__main__':
    # img = cv.imread("/Users/xxh/projects/python/ml/6/strawberries-RGB.tif")
    # cv.imshow("img", img)
    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # row, col = 3, 3
    # plt.subplot(row, col, 1)
    # plt.title("raw", y=-0.2)
    # plt.xticks([])  # 去掉x轴
    # plt.yticks([])  # 去掉y轴
    # plt.axis('off')  # 去掉坐标轴
    # plt.imshow(img)
    #
    # plt.subplot(row, col, 2)
    # plt.title("rgb to cmy", y=-0.2)
    # cmy = rgb_to_cmy(img)
    # plt.xticks([])  # 去掉x轴
    # plt.yticks([])  # 去掉y轴
    # plt.axis('off')  # 去掉坐标轴
    # plt.imshow(cmy)
    #
    # plt.subplot(row, col, 3)
    # cmyk = cmy_to_cmyk(cmy)
    # plt.title("cmy to cmyk", y=-0.2)
    # plt.xticks([])  # 去掉x轴
    # plt.yticks([])  # 去掉y轴
    # plt.axis('off')  # 去掉坐标轴
    # plt.imshow(cmyk)
    #
    # plt.subplot(row, col, 4)
    # plt.title("cmyk to cmy", y=-0.2)
    # plt.xticks([])  # 去掉x轴
    # plt.yticks([])  # 去掉y轴
    # plt.axis('off')  # 去掉坐标轴
    # plt.imshow(cmyk_to_cmy(cmyk))
    #
    # plt.subplot(row, col, 5)
    # plt.title("rgb to hsi", y=-0.2)
    # hsi = rgb_to_hsi(img)
    # plt.xticks([])  # 去掉x轴
    # plt.yticks([])  # 去掉y轴
    # plt.axis('off')  # 去掉坐标轴
    # plt.imshow(hsi)
    #
    # plt.subplot(row, col, 6)
    # plt.title("hsi to rgb", y=-0.2)
    # plt.xticks([])  # 去掉x轴
    # plt.yticks([])  # 去掉y轴
    # plt.axis('off')  # 去掉坐标轴
    # plt.imshow(hsi_to_rgb(hsi))

    # plt.subplot(row, col, 7)
    # plt.title("H", y=-0.2)
    # plt.xticks([])  # 去掉x轴
    # plt.yticks([])  # 去掉y轴
    # plt.axis('off')  # 去掉坐标轴
    # plt.imshow(hsi[:, :, 0] * 255, 'gray')
    #
    # plt.subplot(row, col, 8)
    # plt.title("S", y=-0.2)
    # plt.xticks([])  # 去掉x轴
    # plt.yticks([])  # 去掉y轴
    # plt.axis('off')  # 去掉坐标轴
    # plt.imshow(hsi[:, :, 1] * 255, 'gray')
    #
    # plt.subplot(row, col, 9)
    # plt.title("I", y=-0.2)
    # plt.xticks([])  # 去掉x轴
    # plt.yticks([])  # 去掉y轴
    # plt.axis('off')  # 去掉坐标轴
    # plt.imshow(hsi[:, :, 2] * 255, 'gray')
    #
    # plt.show()
    # img = cv.imread("/Users/xxh/projects/python/ml/3/Fig0316(3)(third_from_top).tif", 0)
    # res = gray_to_colorful(img)
    # cv.imshow("gray_to_colorful", res[...,::-1])
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    img = cv.imread("/Users/xxh/projects/python/ml/6/lenna-RGB.tif")
    cv.imshow("img", img)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # res = colorful_hist_equalization(img).astype(np.uint8)
