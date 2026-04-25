import cv2
import numpy as np

path = '/Users/vishrutgrover/coding/sem6/cv/lab6/'
img = cv2.imread(path + 'dice.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], np.float32)
gy = np.array([[-1,-1,-1],[0,0,0],[1,1,1]], np.float32)
px = cv2.filter2D(gray, cv2.CV_64F, gx)
py = cv2.filter2D(gray, cv2.CV_64F, gy)
prewitt = np.sqrt(px**2 + py**2).astype(np.uint8)
cv2.imwrite(path + '1_prewitt.png', prewitt)

harris = cv2.cornerHarris(gray, 2, 3, 0.04)
harris = cv2.normalize(harris, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imwrite(path + '2_harris.png', harris)

_, global_t = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv2.imwrite(path + '3a_global.png', global_t)
mean_t = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imwrite(path + '3b_adaptive_mean.png', mean_t)
gauss_t = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imwrite(path + '3c_adaptive_gauss.png', gauss_t)