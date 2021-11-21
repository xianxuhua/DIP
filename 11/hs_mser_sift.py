import cv2 as cv


def HS():
    img = cv.imread("/Users/xxh/projects/python/ml/10/building-600by600.tif")
    cv.imshow("img", img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    hs = cv.cornerHarris(gray, 2, 3, 0.04)
    hs = cv.dilate(hs, None)
    img[hs > 0.01 * hs.max()] = [0, 255, 0]
    cv.imshow("hs", img)


def SIFT():
    img = cv.imread("/Users/xxh/projects/python/ml/10/building-600by600.tif")
    sift: cv.SIFT = cv.SIFT_create()
    cv.imshow('sift', cv.drawKeypoints(img, sift.detect(img), img))


def MSER():
    img = cv.imread("/Users/xxh/projects/python/ml/10/building-600by600.tif")
    mser: cv.MSER = cv.MSER_create()
    cv.imshow('mser', cv.drawKeypoints(img, mser.detect(img), img))


if __name__ == '__main__':
    MSER()
    SIFT()
    HS()
    cv.waitKey(0)
    cv.destroyAllWindows()
