import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

try:
    from urllib.request import urlopen, Request
except Exception:
    from urllib2 import urlopen, Request

path = '/Users/vishrutgrover/coding/sem6/cv/lab13/'
os.makedirs(path, exist_ok=True)

def dl(url, p):
    r = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    data = urlopen(r, timeout=60).read()
    with open(p, 'wb') as f:
        f.write(data)

img_url = 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/horses.jpg'
img_path = os.path.join(path, 'img.jpg')
if not os.path.exists(img_path):
    dl(img_url, img_path)

proto_url = 'https://raw.githubusercontent.com/CSAILVision/places365/master/deploy_googlenet_places365.prototxt'
proto_path = os.path.join(path, 'deploy_googlenet_places365.prototxt')
if not os.path.exists(proto_path):
    dl(proto_url, proto_path)

model_url = 'http://places2.csail.mit.edu/models_places365/googlenet_places365.caffemodel'
model_path = os.path.join(path, 'googlenet_places365.caffemodel')
if not os.path.exists(model_path):
    dl(model_url, model_path)

cats_url = 'https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt'
cats_path = os.path.join(path, 'categories_places365.txt')
if not os.path.exists(cats_path):
    dl(cats_url, cats_path)

with open(cats_path, 'r') as f:
    cats = [l.strip().split(' ')[0][3:] for l in f.readlines()]

img = cv2.imread(img_path)
net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
blob = cv2.dnn.blobFromImage(img, 1.0, (224, 224), (104, 117, 123), swapRB=False, crop=True)
net.setInput(blob)
p = net.forward().flatten()

idx = np.argsort(p)[::-1][:5]
top = [(cats[i], float(p[i])) for i in idx]

out = img.copy()
y = 30
for name, conf in top:
    txt = '{}: {:.2f}'.format(name, conf)
    cv2.rectangle(out, (5, y - 22), (5 + 12 * len(txt), y + 8), (0, 0, 0), -1)
    cv2.putText(out, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y += 32

print('Top-5 predictions (Places365):')
for name, conf in top:
    print('  {:25s} {:.4f}'.format(name, conf))

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax[0].set_title('Input')
ax[0].axis('off')
ax[1].imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
ax[1].set_title('Classification (Places365)')
ax[1].axis('off')
plt.tight_layout()
plt.show()
