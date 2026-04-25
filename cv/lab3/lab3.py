import cv2
import numpy as np

path = '/Users/vishrutgrover/coding/sem6/cv/lab3/'
img = cv2.imread(path + 'waldo.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 1. Bit plane slicing of grayscale image
for i in range(8): cv2.imwrite(path + f'bitplane{i}.jpg', ((gray >> i) & 1) * 255)

# 2. Averaging filter kernels: a. 3x3, b. 5x5, c. 10x15
cv2.imwrite(path + 'avg3x3.jpg', cv2.blur(img, (3, 3)))
cv2.imwrite(path + 'avg5x5.jpg', cv2.blur(img, (5, 5)))
cv2.imwrite(path + 'avg10x15.jpg', cv2.blur(img, (10, 15)))

# 3. Gaussian Smoothing
cv2.imwrite(path + 'gaussian.jpg', cv2.GaussianBlur(img, (5, 5), 0))

# 4. Contrast Stretching of grayscale image
min_val, max_val = gray.min(), gray.max()
stretched = ((gray - min_val) / (max_val - min_val) * 255).astype(np.uint8)
cv2.imwrite(path + 'contrast.jpg', stretched)