import numpy as np
import cv2 as cv


# 标准化矩阵
standard_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])


def block_dct_compress(gray_img):
    block_size = 8
    new_height = gray_img.shape[0] - gray_img.shape[0] % block_size
    new_width = gray_img.shape[1] - gray_img.shape[1] % block_size
    gray_img = cv.resize(gray_img, [new_height, new_width])
    block_matrix = np.zeros([new_height // block_size, new_width // block_size], np.ndarray)
    # 转为8*8的子图像
    for x in range(block_matrix.shape[0]):
        for y in range(block_matrix.shape[1]):
            block_matrix[x, y] = gray_img[x*block_size: (x + 1)*block_size, y*block_size: (y + 1) * block_size]
            block_matrix[x, y] -= 128
            # 子图像dct
            block_matrix[x, y] = cv.dct(block_matrix[x, y].astype(np.float32))
            # 量化
            block_matrix[x, y] = np.round(block_matrix[x, y] / standard_matrix)
            # 去规格化
            block_matrix[x, y] *= standard_matrix
            block_matrix[x, y] = cv.idct(block_matrix[x, y])
            block_matrix[x, y] += 128

    # 图像重建
    temp_matrix = np.zeros([1, block_matrix.shape[1]], np.ndarray)
    for i in range(block_matrix.shape[0]):
        temp_matrix[0, i] = np.vstack(block_matrix[:, i])
    recovered_img = np.hstack(temp_matrix[0, :])

    return recovered_img


if __name__ == '__main__':
    img = cv.imread("/Users/xxh/projects/python/ml/5/book-cover.tif", 0)
    cv.imshow("img", img)
    res = block_dct_compress(img).astype(np.uint8)
    cv.imshow("res.jpg", res)
    cv.waitKey(0)
    cv.destroyAllWindows()
