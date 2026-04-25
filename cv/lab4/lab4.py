# Exp 4: Median filter, Sharpening, Histogram equalization, Edge detection
import cv2
import numpy as np

path = '/Users/vishrutgrover/coding/sem6/cv/lab4/'
img = cv2.imread(path + 'dice.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 1. Median filtering
cv2.imwrite(path + '1_median.png', cv2.medianBlur(img, 5))

# 2. Image sharpening (kernels a & b)
ka = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]], np.float32)
kb = np.array([[1,1,1],[1,-7,1],[1,1,1]], np.float32)
cv2.imwrite(path + '2a_sharpen.png', cv2.filter2D(img, -1, ka))
cv2.imwrite(path + '2b_sharpen.png', cv2.filter2D(img, -1, kb))

# 3. Histogram equalization
cv2.imwrite(path + '3_histeq.png', cv2.equalizeHist(gray))

# 4. Edge detection (derivative filter - Sobel)
gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
edges = np.sqrt(gx**2 + gy**2).astype(np.uint8)
cv2.imwrite(path + '4_edges.png', edges)