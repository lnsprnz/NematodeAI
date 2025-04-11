import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
 
img = cv.imread(r'NematodeAI\Data\Images DeadLiveCounting\feste Kamera 1\feste Kamera 1_frame_0.png')
assert img is not None, "file could not be read, check with os.path.exists()"
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
# sure background area
sure_bg = cv.dilate(opening,kernel,iterations=3)

cv.imshow('Thresholded Image', sure_bg)
cv.waitKey(0)
cv.destroyAllWindows()