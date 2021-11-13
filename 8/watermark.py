import numpy as np
import cv2 as cv


def add_simple_watermark(grey_img, copy_right_img, alpha):
    height, width = copy_right_img.shape
    grey_img[:height, :width] = (1 - alpha) * grey_img[:height, :width] + alpha * copy_right_img
    return grey_img


def add_no_found_watermark(grey_img, copy_right_img, k, alpha):
    added_img = add_simple_watermark(grey_img, copy_right_img, alpha)
    added_dct: np.ndarray = cv.dct(added_img.astype(np.float32))
    dim1_max_index = added_dct.flatten().argsort()[::-1][0:k]
    dim2_max_x = dim1_max_index // copy_right_img.shape[0]
    dim2_max_y = dim1_max_index % copy_right_img.shape[0]
    w = np.random.normal(0, 1, k)
    for index, (x, y) in enumerate(list(zip(dim2_max_x, dim2_max_y))):
        added_dct[x, y] = added_dct[x, y] * (1 + alpha * w[index])

    return cv.idct(added_dct)


if __name__ == '__main__':
    img = cv.imread("/Users/xxh/projects/python/ml/5/book-cover.tif", 0)
    copy_right_img = cv.imread("/Users/xxh/projects/python/ml/2/shop_qrcode.jpg", 0)
    res = add_no_found_watermark(img, copy_right_img, 5, 0.01).astype(np.uint8)
    cv.imshow("res", res)
    cv.waitKey(0)
    cv.destroyAllWindows()
