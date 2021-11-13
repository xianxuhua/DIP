import pywt
import numpy as np
import cv2 as cv


def threshold(img_arr, th):
    return np.where(np.abs(img_arr) > th, img_arr, 0)


def wavelet_compress(grey_img, th):
    # 二级压缩
    # A2, D2, D1 = pywt.wavedec(grey_img, 'haar', level=2)
    # A2, D2, D1 = threshold(A2, th), threshold(D2, th), threshold(D1, th)
    # return pywt.waverec([A2, D2, D1], 'haar').astype(np.uint8)
    # 一级压缩
    A, (H, V, D) = pywt.dwt2(grey_img, 'haar')
    A, H, V, D = threshold(A, th), threshold(H, th), threshold(V, th), threshold(D, th)
    return pywt.idwt2((A, (H, V, D)), 'haar').astype(np.uint8)


if __name__ == '__main__':
    img = cv.imread("/Users/xxh/projects/python/ml/5/book-cover.tif", 0)
    cv.imshow("img", img)
    res = wavelet_compress(img, 100)
    cv.imwrite("res.jpg", res)
    cv.waitKey(0)
    cv.destroyAllWindows()
