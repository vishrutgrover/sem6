# Exp 8: To compute the Mean, Standard Deviation, and Correlation Coefficient of a given image.

import cv2
import numpy as np

path = '/Users/vishrutgrover/coding/sem6/dip/lab8/'
img = cv2.imread(path + 'cube.jpeg')

if img is None:
    raise FileNotFoundError("Image not found. Ensure cube.jpeg exists in lab8 folder.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

mean = np.mean(gray)
std = np.std(gray)

pixels = gray.flatten().astype(np.float64)
pixel_left = pixels[:-1]
pixel_right = pixels[1:]
corr_coef = np.corrcoef(pixel_left, pixel_right)[0, 1]

print("Grayscale Image Statistics:")
print("  Mean: {:.4f}".format(mean))
print("  Standard Deviation: {:.4f}".format(std))
print("  Correlation Coefficient (horizontal neighbors): {:.4f}".format(corr_coef))

print("\nPer-Channel Statistics (BGR):")
for i, name in enumerate(['Blue', 'Green', 'Red']):
    ch = img[:, :, i].astype(np.float64)
    ch_mean = np.mean(ch)
    ch_std = np.std(ch)
    print("  {}: Mean = {:.4f}, Std = {:.4f}".format(name, ch_mean, ch_std))

b, g, r = img[:, :, 0].flatten(), img[:, :, 1].flatten(), img[:, :, 2].flatten()
corr_bg = np.corrcoef(b, g)[0, 1]
corr_br = np.corrcoef(b, r)[0, 1]
corr_gr = np.corrcoef(g, r)[0, 1]
print("\nChannel Correlation Coefficients:")
print("  Blue-Green: {:.4f}".format(corr_bg))
print("  Blue-Red:   {:.4f}".format(corr_br))
print("  Green-Red:  {:.4f}".format(corr_gr))