# Exp 7: To compute and display the 1-D and 2-D Fast Fourier Transform (FFT) of a grayscale image.

import cv2
import numpy as np
import matplotlib.pyplot as plt

path = '/Users/vishrutgrover/coding/sem6/dip/lab7/'
img = cv2.imread(path + 'head.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

row = gray[gray.shape[0]//2, :].astype(np.float32)
fft1d = np.fft.fft(row)
mag1d = np.abs(fft1d)

fft2d = np.fft.fft2(gray.astype(np.float32))
fft2d = np.fft.fftshift(fft2d)
mag2d = np.log(1 + np.abs(fft2d))

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(gray, cmap='gray')
ax[0].set_title('Original')
ax[0].axis('off')
ax[1].plot(mag1d)
ax[1].set_title('1-D FFT')
ax[2].imshow(mag2d, cmap='gray')
ax[2].set_title('2-D FFT')
ax[2].axis('off')
plt.tight_layout()
plt.show()