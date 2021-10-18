import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def sub_hist_equalization_with_statistic(img):
    height, width = img.shape
    k0, k1, k2, k3, C = 0.5, 0.8, 0, 0.2, 30
    m_g = np.mean(img)
    sigama_g = np.var(img)
    s_size = 3
    res = np.zeros((height, width), np.uint8)
    for i in range(height):
        for j in range(width):
            m_s = np.mean(img[i:i+s_size, j:j+s_size])
            sigama_s = np.var(img[i:i+s_size, j:j+s_size])
            if k0 * m_g <= m_s <= k1 * m_g and k2 * sigama_g <= sigama_s <= k3 * sigama_g:
                res[i, j] = C * img[i, j]
            else:
                res[i, j] = img[i, j]

    return res


def bounds225(res):
    mi, ma = np.min(res), np.max(res)
    res -= mi
    return 255 * (res / (ma - mi))


def phase_restructure(img):
    return bounds225(np.abs(np.fft.ifft2(np.exp(1j * np.angle(np.fft.fft2(img))))))


def amplitude_restructure(img):
    return bounds225(np.abs(np.fft.ifft2(np.abs(np.fft.fft2(img)))))


if __name__ == '__main__':
    img = cv.imread("/Users/xxh/projects/python/ml/4/boy.tif", 0)
    mask = cv.imread("/Users/xxh/projects/python/ml/4/wingding-square-solid.tif", 0)

    # np.multiply(np.abs(f), np.exp(1j*np.angle(f2)))

    # mask.resize((600, 688))
    # mask.resize((688, 600))
    # mask_fft = np.abs(np.fft.fft2(mask))
    # img_fft = np.angle(np.fft.fft2(img))
    # res = np.fft.ifft2(np.multiply(img_fft, np.exp(1j * mask_fft)))
    # print(bounds225(np.abs(res)))

    cv.imwrite("res.png",
        phase_restructure(img))
    print("success")
    cv.waitKey(0)
    cv.destroyAllWindows()
    # plt.imshow(res, 'gray')

    # plt.subplot(2, 2, 1)
    # plt.imshow(img, 'gray')
    # plt.subplot(2, 2, 2)
    # plt.imshow(bounds225(np.abs(np.fft.fft2(img))), 'gray')
    # plt.subplot(2, 2, 3)
    # plt.imshow(bounds225(np.fft.fftshift(np.abs(np.fft.fft2(img)))), 'gray')
    # plt.subplot(2, 2, 4)
    # plt.imshow(bounds225(np.fft.fftshift(np.log(1 + np.abs(np.fft.fft2(img))))), 'gray')
    plt.show()
    print("success")
