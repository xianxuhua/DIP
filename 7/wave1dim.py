import numpy as np
import matplotlib.pyplot as plt
import pywt
from skimage.restoration import denoise_wavelet


def decompose():
    x = np.arange(0, 10)
    y = np.sin(x)
    # cA, cD = pywt.dwt(y, 'haar')
    # y2 = pywt.idwt(cA, cD, 'haar')


def multi_decompose():
    x = np.arange(0, 10)
    y = np.sin(x)
    # 多层分解
    #     cD1
    #   /
    # s       cD2
    #   \   /
    #    cA1
    #      \
    #       cA2
    cA2, cD2, cD1 = pywt.wavedec(y, 'haar', level=2)
    print(cA2)
    print(cD1)
    print(cD2)
    y3 = pywt.waverec([cA2, cD2, cD1], 'haar')
    print(y3)


def remove_noise():
    x = pywt.data.ecg().astype(float) / 256
    sigma = 0.05
    x += sigma * np.random.randn(x.size)
    denoised_x = denoise_wavelet(x, sigma, wavelet='haar', mode='soft', wavelet_levels=3, method='BayesShrink')
    plt.figure(figsize=[10, 8])
    plt.plot(x)
    plt.plot(denoised_x)
    plt.show()


if __name__ == '__main__':
    remove_noise()
