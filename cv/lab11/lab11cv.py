import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

try:
    from urllib.request import urlretrieve
except Exception:
    from urllib import urlretrieve

path = '/Users/vishrutgrover/coding/sem6/cv/lab11/'
os.makedirs(path, exist_ok=True)
url = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/digits.png'
img_path = os.path.join(path, 'digits.png')
if not os.path.exists(img_path):
    urlretrieve(url, img_path)

img = cv2.imread(img_path, 0)
cells = [np.hsplit(r, 100) for r in np.vsplit(img, 50)]
x = np.array(cells).reshape(-1, 20, 20)
y = np.repeat(np.arange(10), 500).astype(np.int32)

idx = np.arange(x.shape[0])
np.random.seed(0)
np.random.shuffle(idx)
x = x[idx]
y = y[idx]

n = int(0.8 * x.shape[0])
xtr, xte = x[:n], x[n:]
ytr, yte = y[:n], y[n:]

xtr = xtr.reshape(-1, 400).astype(np.float32)
xte = xte.reshape(-1, 400).astype(np.float32)

knn = cv2.ml.KNearest_create()
knn.train(xtr, cv2.ml.ROW_SAMPLE, ytr)
_, res, _, _ = knn.findNearest(xte, k=5)
pred = res.reshape(-1).astype(np.int32)
acc = float(np.mean(pred == yte))

show = 12
im = xte[:show].reshape(show, 20, 20)
tt = yte[:show]
pp = pred[:show]
grid = np.hstack(im)

print('kNN accuracy: {:.4f}'.format(acc))
print('true:', tt.tolist())
print('pred:', pp.tolist())

fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].imshow(grid, cmap='gray')
ax[0].set_title('Test samples')
ax[0].axis('off')
ax[1].bar(['kNN'], [acc])
ax[1].set_ylim(0, 1)
ax[1].set_title('Accuracy')
plt.tight_layout()
plt.show()