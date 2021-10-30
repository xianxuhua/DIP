import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def bounds225(res):
    return np.uint8(255 * normalization(res))


def normalization(res):
    mi, ma = np.min(res), np.max(res)
    res -= mi
    return res / (ma - mi)


def turbulence_degeneration_func(img, k):
    M, N = img.shape
    H = np.zeros(img.shape)
    for u in range(M):
        for v in range(N):
            H[u, v] = np.exp(-k * np.power((u-M//2)**2+(v-N//2)**2, 5/6))

    return H


def move_degeneration_func(img, T, a, b):
    M, N = img.shape
    H = np.zeros(img.shape, np.complex64)
    center_x = M // 2
    center_y = N // 2
    for u in range(M):
        for v in range(N):
            t = np.pi * ((u-center_x) * a + (v-center_y) * b)
            H[u, v] = T / (t+0.001) * np.sin(t) * np.exp(-1j * t)

    return H


def degenerate(img, H):
    F = np.fft.fftshift(np.fft.fft2(img))
    return bounds225(np.real(np.fft.ifft2(np.fft.ifftshift(F * H))))


# threshold: 截止频率
def inverse(noise_img, H, threshold):
    G = np.fft.fftshift(np.fft.fft2(noise_img))
    M, N = noise_img.shape
    F = np.zeros(noise_img.shape, np.complex64)
    for i in range(M):
        for j in range(N):
            if np.sqrt(np.power(i-M//2, 2) + np.power(j-N//2, 2)) < threshold:
                # 避免除以0
                F[i, j] = G[i, j] / (H[i, j] + 0.001)

    res = np.real(np.fft.ifft2(np.fft.ifftshift(F)))
    return bounds225(res)


def wiener_filter(noise_img, H, K):
    G = np.fft.fftshift(np.fft.fft2(noise_img))
    F = np.conj(H) / (np.power(np.abs(H), 2) + K) * G
    return bounds225(np.real(np.fft.ifft2(np.fft.ifftshift(F))))


def constrained_least_squares(noise_img, H, noise_mean, noise_var):
    G = np.fft.fftshift(np.fft.fft2(noise_img))
    gama = 0.00001
    alpha = 0.25
    step = 0.000001
    eta_2 = noise_img.shape[0] * noise_img.shape[1] * (noise_var + noise_mean ** 2)
    while True:
        if gama ** 2 < (eta_2 - alpha):
            gama += step
        elif gama ** 2 > (eta_2 + alpha):
            gama -= step
        else:
            break

    P = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0],
    ])
    pad_m_top = (H.shape[0] - 3 + 1) // 2
    pad_m_bottom = (H.shape[0] - 3) // 2
    pad_m_left = (H.shape[1] - 3 + 1) // 2
    pad_m_right = (H.shape[1] - 3) // 2
    P = cv.copyMakeBorder(P, pad_m_top, pad_m_bottom, pad_m_left, pad_m_right, cv.BORDER_CONSTANT)
    F = np.conj(H) / (0.001 + np.power(np.abs(H), 2) + gama * np.power(P, 2)) * G
    return bounds225(np.real(np.fft.ifft2(np.fft.ifftshift(F))))


def gauss_noise(img, mu, sigma):
    noise = np.random.normal(mu, sigma, img.shape)
    res = img + noise
    return np.uint8(bounds225(res)), noise


if __name__ == '__main__':
    img = cv.imread("/Users/xxh/projects/python/ml/5/aerial_view.tif", 0)
    cv.imshow("img", img)
    # H = turbulence_degeneration_func(img, 0.0025)
    H = move_degeneration_func(img, 1, 0.1, 0.1)
    noise_img = degenerate(img, H)
    noise_img, noise = gauss_noise(noise_img, 0, np.sqrt(0.00001))

    # plt.subplot(2, 2, 1)
    # plt.imshow(noise_img, 'gray')
    # plt.subplot(2, 2, 2)
    # plt.imshow(inverse(noise_img, H, 40), 'gray')
    # plt.subplot(2, 2, 3)
    # plt.imshow(inverse(noise_img, H, 100), 'gray')
    # plt.subplot(2, 2, 4)
    # plt.imshow(inverse(noise_img, H, 200), 'gray')
    # plt.show()
    cv.imshow("noise_img", noise_img)
    cv.imshow("inverse", inverse(noise_img, H, 100))
    cv.imshow("wiener_filter", wiener_filter(noise_img, H, 0.003))
    cv.imshow("constrained_least_squares", constrained_least_squares(noise_img, H, 0, 0.00001))
    print("success")
    cv.waitKey(0)
    cv.destroyAllWindows()
