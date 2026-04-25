# Exp 5: Morphology, Laplacian, Canny, Thresholding
import cv2
import numpy as np

path = '/Users/vishrutgrover/coding/sem6/cv/lab5/'
img = cv2.imread(path + 'gojo.webp')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
k = np.ones((5, 5), np.uint8)

# 1. Morphological transformations
cv2.imwrite(path + '1a_erosion.png', cv2.erode(binary, k))
cv2.imwrite(path + '1b_dilation.png', cv2.dilate(binary, k))
cv2.imwrite(path + '1c_opening.png', cv2.morphologyEx(binary, cv2.MORPH_OPEN, k))
cv2.imwrite(path + '1d_closing.png', cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k))
cv2.imwrite(path + '1e_gradient.png', cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, k))

# 2. Laplacian edge detection (kernel [0 1 0; 1 -4 1; 0 1 0])
cv2.imwrite(path + '2_laplacian.png', cv2.Laplacian(gray, cv2.CV_64F, ksize=1))

# 3. Canny edge detection
cv2.imwrite(path + '3_canny.png', cv2.Canny(gray, 100, 200))

# 4. Image thresholding (all 5 types)
for name, thresh_type in [('binary', cv2.THRESH_BINARY), ('binary_inv', cv2.THRESH_BINARY_INV),
                         ('trunc', cv2.THRESH_TRUNC), ('tozero', cv2.THRESH_TOZERO),
                         ('tozero_inv', cv2.THRESH_TOZERO_INV)]:
    _, out = cv2.threshold(gray, 127, 255, thresh_type)
    cv2.imwrite(path + f'4_{name}.png', out)
