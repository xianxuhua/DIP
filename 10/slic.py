import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
import cv2 as cv

# img = cv.imread("/Users/xxh/projects/python/ml/6/lenna-RGB.tif")
img = cv.imread("/Users/xxh/projects/python/ml/10/iceberg.tif")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
segments = slic(img, n_segments=100)
plt.subplot(131)
plt.title('image')
plt.imshow(img, 'gray')
plt.subplot(132)
plt.title('segments')
plt.imshow(segments, 'gray')
plt.subplot(133)
plt.title('image and segments')
plt.imshow(mark_boundaries(img, segments), 'gray')
plt.show()
