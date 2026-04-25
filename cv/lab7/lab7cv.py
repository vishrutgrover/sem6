import cv2
import numpy as np
import matplotlib.pyplot as plt

path = '/Users/vishrutgrover/coding/sem6/cv/lab7/'
img = cv2.imread(path + 'messi.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

pixels = img.reshape(-1, 3).astype(np.float32)
_, labels, centers = cv2.kmeans(pixels, 3, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 5, cv2.KMEANS_RANDOM_CENTERS)
kmeans_img = centers[labels.flatten().astype(int)].reshape(img.shape).astype(np.uint8)

edges = cv2.Canny(gray, 50, 150)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=50, maxLineGap=10)
hough_img = img.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(hough_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

edge_seg = cv2.Canny(gray, 50, 150)

_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
num_labels, labels = cv2.connectedComponents(thresh)
region_seg = np.uint8(255 * labels / max(labels.max(), 1))

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
boxes, _ = hog.detectMultiScale(img, winStride=(8, 8), padding=(32, 32), scale=1.05)
hog_img = img.copy()
for (x, y, w, h) in boxes:
    cv2.rectangle(hog_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

fig, ax = plt.subplots(2, 3, figsize=(12, 8))
ax[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax[0, 0].set_title('Original')
ax[0, 0].axis('off')
ax[0, 1].imshow(cv2.cvtColor(kmeans_img, cv2.COLOR_BGR2RGB))
ax[0, 1].set_title('K-Means k=3')
ax[0, 1].axis('off')
ax[0, 2].imshow(cv2.cvtColor(hough_img, cv2.COLOR_BGR2RGB))
ax[0, 2].set_title('Hough Lines')
ax[0, 2].axis('off')
ax[1, 0].imshow(edge_seg, cmap='gray')
ax[1, 0].set_title('Edge based')
ax[1, 0].axis('off')
ax[1, 1].imshow(region_seg, cmap='gray')
ax[1, 1].set_title('Region based')
ax[1, 1].axis('off')
ax[1, 2].imshow(cv2.cvtColor(hog_img, cv2.COLOR_BGR2RGB))
ax[1, 2].set_title('HOG Detection')
ax[1, 2].axis('off')
plt.tight_layout()
plt.show()