# 1. HSV conversion 2. Crop image 3. Blur with kernel 4. Find contours
import cv2
import numpy as np
from PIL import Image

path = '/Users/vishrutgrover/coding/sem6/cv/lab2/'
img = cv2.imread(path + 'claw.png')

# 1. HSV conversion
cv2.imwrite(path + 'hsvclaw.png', cv2.cvtColor(img, cv2.COLOR_BGR2HSV))

# 2. Crop image
Image.open(path + 'claw.png').crop((300, 100, 600, 400)).save(path + 'cropclaw.png')

# 3. Blur with kernel
cv2.imwrite(path + 'blurredclaw.png', cv2.filter2D(img, -1, np.ones((5, 5), np.float32) / 25))

# 4. Find contours
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
contours, _ = cv2.findContours(cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
cv2.imwrite(path + 'contoursclaw.png', img)

# 5. Text on image
cv2.putText(img, 'vishrut', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
cv2.imwrite(path + 'vishrutclaw.png', img)