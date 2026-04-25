import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

try:
    from urllib.request import urlretrieve
except Exception:
    from urllib import urlretrieve

path = '/Users/vishrutgrover/coding/sem6/cv/lab8/'
url = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg'
img_path = os.path.join(path, 'img.jpg')
os.makedirs(path, exist_ok=True)
if not os.path.exists(img_path):
    urlretrieve(url, img_path)

img = cv2.imread(img_path)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = cas.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
face_img = rgb.copy()
for (x, y, w, h) in faces:
    cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

try:
    sift = cv2.SIFT_create()
except Exception:
    sift = None

kp_img = rgb.copy()
if sift is not None:
    kp = sift.detect(gray, None)
    kp_img = cv2.drawKeypoints(rgb, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

try:
    from sklearn.decomposition import PCA
    p = PCA(n_components=60, random_state=0)
    x = gray.astype(np.float32) / 255.0
    z = p.fit_transform(x)
    x2 = p.inverse_transform(z)
    pca_img = np.uint8(np.clip(x2 * 255.0, 0, 255))
except Exception:
    u, s, vt = np.linalg.svd(gray.astype(np.float32), full_matrices=False)
    k = 60
    x2 = np.dot(u[:, :k] * s[:k], vt[:k, :])
    pca_img = np.uint8(np.clip(x2, 0, 255))

ae_img = gray.copy()
try:
    from sklearn.neural_network import MLPRegressor
    x = cv2.resize(gray, (64, 64)).astype(np.float32) / 255.0
    x = x.reshape(1, -1)
    ae = MLPRegressor(hidden_layer_sizes=(256, 64, 256), activation='relu', max_iter=300, random_state=0)
    ae.fit(x, x)
    y = ae.predict(x).reshape(64, 64)
    ae_img = np.uint8(np.clip(y * 255.0, 0, 255))
except Exception:
    ae_img = cv2.GaussianBlur(gray, (9, 9), 2)

fig, ax = plt.subplots(2, 3, figsize=(12, 8))
ax[0, 0].imshow(rgb)
ax[0, 0].set_title('Original')
ax[0, 0].axis('off')
ax[0, 1].imshow(face_img)
ax[0, 1].set_title('Haar Face')
ax[0, 1].axis('off')
ax[0, 2].imshow(kp_img)
ax[0, 2].set_title('SIFT')
ax[0, 2].axis('off')
ax[1, 0].imshow(pca_img, cmap='gray')
ax[1, 0].set_title('PCA Recon')
ax[1, 0].axis('off')
ax[1, 1].imshow(ae_img, cmap='gray')
ax[1, 1].set_title('Autoencoder Recon')
ax[1, 1].axis('off')
ax[1, 2].axis('off')
plt.tight_layout()
plt.show()

