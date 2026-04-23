import cv2
import numpy as np
import matplotlib.pyplot as plt

path = '/Users/vishrutgrover/coding/sem6/dip/lab11/'
img = cv2.imread(path + 'ex3.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

can = cv2.Canny(gray, 100, 200)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(gray, cmap='gray')
ax[0].set_title('Original')
ax[0].axis('off')
ax[1].imshow(can, cmap='gray')
ax[1].set_title('Canny Edges')
ax[1].axis('off')
plt.tight_layout()
plt.show()