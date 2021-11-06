import numpy as np
import cv2 as cv


def compress(gray_img, offset):
    img_dct = cv.dct(gray_img.astype(np.float32))
    height, width = gray_img.shape
    for x in range(height):
        for y in range(width):
            if x > height * offset or y > width * offset:
                img_dct[x, y] = 0

    cv.imshow("img_dct", np.log(np.abs(img_dct)).astype(np.uint8))
    return cv.idct(img_dct)


if __name__ == '__main__':
    img = cv.imread("/Users/xxh/projects/python/ml/6/strawberries-RGB.tif")
    cv.imwrite("img.jpg", cv.cvtColor(img, cv.COLOR_BGR2GRAY))
    cv.imwrite("res.jpg", compress(cv.cvtColor(img, cv.COLOR_BGR2GRAY), 0.4).astype(np.uint8), params=[cv.IMWRITE_JPEG_QUALITY, 30])
    cv.imshow("crcb", cv.cvtColor(img, cv.COLOR_RGB2YCR_CB).astype(np.uint8))
    cv.waitKey(0)
    cv.destroyAllWindows()
