import cv2
import numpy as np
img = cv2.imread('/Users/vishrutgrover/coding/sem6/cv/lab3/waldo.jpg')
kernel_size = 5
kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
blurred = cv2.filter2D(img, -1, kernel)
cv2.imwrite('/Users/vishrutgrover/coding/sem6/cv/lab3/blurwaldo.jpg', blurred)
print("Image blurred and saved as 'blurwaldo.jpg'")