import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

try:
    from urllib.request import urlretrieve
except Exception:
    from urllib import urlretrieve

path = '/Users/vishrutgrover/coding/sem6/cv/lab9/'
url = 'https://picsum.photos/640/480'
img_path = os.path.join(path, 'img.jpg')
os.makedirs(path, exist_ok=True)
if not os.path.exists(img_path):
    urlretrieve(url, img_path)

img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
g = cv2.resize(gray, (160, 120))
h, w = g.shape

_, y = cv2.threshold(g, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
yy, xx = np.indices((h, w))
x = np.stack([xx, yy, g], axis=-1).reshape(-1, 3).astype(np.float32)
y = y.reshape(-1).astype(np.int32)

np.random.seed(0)
idx = np.random.choice(x.shape[0], min(2000, x.shape[0]), replace=False)
xtr, ytr = x[idx], y[idx]

def knn_pred(xtr, ytr, xq, k=5, bs=2048):
    b = np.sum(xtr * xtr, axis=1)
    out = np.zeros((xq.shape[0],), np.uint8)
    i = 0
    while i < xq.shape[0]:
        j = min(i + bs, xq.shape[0])
        q = xq[i:j]
        a = np.sum(q * q, axis=1, keepdims=True)
        d = a + b - 2 * np.dot(q, xtr.T)
        nn = np.argpartition(d, k - 1, axis=1)[:, :k]
        out[i:j] = (np.mean(ytr[nn], axis=1) >= 0.5).astype(np.uint8)
        i = j
    return out

yk = knn_pred(xtr, ytr, x, k=7).reshape(h, w) * 255

try:
    from sklearn.svm import SVC
    s1 = SVC(kernel='linear', C=1.0)
    s1.fit(xtr, ytr)
    yl = s1.predict(x).reshape(h, w).astype(np.uint8) * 255
    s2 = SVC(kernel='rbf', C=2.0, gamma='scale')
    s2.fit(xtr, ytr)
    yr = s2.predict(x).reshape(h, w).astype(np.uint8) * 255
except Exception:
    yl = yk.copy()
    yr = yk.copy()

fig, ax = plt.subplots(2, 2, figsize=(10, 8))
ax[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax[0, 0].set_title('Image')
ax[0, 0].axis('off')
ax[0, 1].imshow((y.reshape(h, w) * 255).astype(np.uint8), cmap='gray')
ax[0, 1].set_title('Labels (Otsu)')
ax[0, 1].axis('off')
ax[1, 0].imshow(yk, cmap='gray')
ax[1, 0].set_title('KNN')
ax[1, 0].axis('off')
ax[1, 1].imshow(yl, cmap='gray')
ax[1, 1].set_title('SVM (linear)')
ax[1, 1].axis('off')
plt.tight_layout()
plt.show()