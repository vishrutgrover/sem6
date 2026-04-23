import cv2
import numpy as np
import matplotlib.pyplot as plt

path = '/Users/vishrutgrover/coding/sem6/dip/lab9/'
img = cv2.imread(path + 'chess.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

noise = np.random.normal(0, 25, gray.shape).astype(np.float32)
noisy = np.clip(gray.astype(np.float32) + noise, 0, 255).astype(np.uint8)

salt_pepper = np.random.random(gray.shape)
noisy_sp = gray.copy()
noisy_sp[salt_pepper < 0.05] = 0
noisy_sp[salt_pepper > 0.95] = 255

mean_filtered = cv2.blur(noisy_sp, (5, 5))
median_filtered = cv2.medianBlur(noisy_sp, 5)
gaussian_filtered = cv2.GaussianBlur(noisy_sp, (5, 5), 1.0)

kernel = np.ones((5, 5), np.float32) / 25
conv_filtered = cv2.filter2D(noisy_sp, -1, kernel)

def mse(a, b):
    return np.mean((a.astype(float) - b.astype(float)) ** 2)

def psnr(a, b):
    m = mse(a, b)
    if m == 0: return float('inf')
    return 10 * np.log10(255**2 / m)

print("Noise Reduction Analysis (Salt-Pepper Noise):")
print("  Original vs Noisy - MSE: {:.2f}, PSNR: {:.2f} dB".format(mse(gray, noisy_sp), psnr(gray, noisy_sp)))
print("  Mean Filter      - MSE: {:.2f}, PSNR: {:.2f} dB".format(mse(gray, mean_filtered), psnr(gray, mean_filtered)))
print("  Median Filter    - MSE: {:.2f}, PSNR: {:.2f} dB".format(mse(gray, median_filtered), psnr(gray, median_filtered)))
print("  Gaussian Filter  - MSE: {:.2f}, PSNR: {:.2f} dB".format(mse(gray, gaussian_filtered), psnr(gray, gaussian_filtered)))
print("  Convolution (5x5)- MSE: {:.2f}, PSNR: {:.2f} dB".format(mse(gray, conv_filtered), psnr(gray, conv_filtered)))

fig, ax = plt.subplots(2, 3, figsize=(12, 8))
ax[0, 0].imshow(gray, cmap='gray')
ax[0, 0].set_title('Original')
ax[0, 0].axis('off')
ax[0, 1].imshow(noisy_sp, cmap='gray')
ax[0, 1].set_title('Noisy (Salt-Pepper)')
ax[0, 1].axis('off')
ax[0, 2].imshow(mean_filtered, cmap='gray')
ax[0, 2].set_title('Mean Filter (5x5)')
ax[0, 2].axis('off')
ax[1, 0].imshow(median_filtered, cmap='gray')
ax[1, 0].set_title('Median Filter (5x5)')
ax[1, 0].axis('off')
ax[1, 1].imshow(gaussian_filtered, cmap='gray')
ax[1, 1].set_title('Gaussian Filter (5x5)')
ax[1, 1].axis('off')
ax[1, 2].imshow(conv_filtered, cmap='gray')
ax[1, 2].set_title('Convolution (5x5 Box)')
ax[1, 2].axis('off')
plt.tight_layout()
plt.show()