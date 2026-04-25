import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

try:
    from urllib.request import urlopen, Request
except Exception:
    from urllib2 import urlopen, Request
import tarfile

path = '/Users/vishrutgrover/coding/sem6/cv/lab12/'
os.makedirs(path, exist_ok=True)

def dl(url, p):
    r = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with open(p, 'wb') as f:
        f.write(urlopen(r, timeout=30).read())

img_url = 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg'
img_path = os.path.join(path, 'img.jpg')
if not os.path.exists(img_path):
    dl(img_url, img_path)

tgz_url = 'http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz'
tgz_path = os.path.join(path, 'faster_rcnn_resnet50_coco_2018_01_28.tar.gz')
if not os.path.exists(tgz_path):
    dl(tgz_url, tgz_path)

pb_path = os.path.join(path, 'frozen_inference_graph.pb')
if not os.path.exists(pb_path):
    with tarfile.open(tgz_path, 'r:gz') as t:
        m = t.getmember('faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb')
        m.name = os.path.basename(m.name)
        t.extract(m, path)

pbtxt_url = 'https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/faster_rcnn_resnet50_coco_2018_01_28.pbtxt'
pbtxt_path = os.path.join(path, 'faster_rcnn_resnet50_coco_2018_01_28.pbtxt')
if not os.path.exists(pbtxt_path):
    dl(pbtxt_url, pbtxt_path)

coco = ['bg','person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']

img = cv2.imread(img_path)
h, w = img.shape[:2]
net = cv2.dnn.readNetFromTensorflow(pb_path, pbtxt_path)
blob = cv2.dnn.blobFromImage(cv2.resize(img, (1024, 600)), swapRB=True, crop=False)
net.setInput(blob)
d = net.forward()

out = img.copy()
if len(d.shape) == 4 and d.shape[3] >= 7:
    for i in range(d.shape[2]):
        conf = float(d[0, 0, i, 2])
        cid = int(d[0, 0, i, 1])
        if conf < 0.25:
            continue
        x1 = int(d[0, 0, i, 3] * w)
        y1 = int(d[0, 0, i, 4] * h)
        x2 = int(d[0, 0, i, 5] * w)
        y2 = int(d[0, 0, i, 6] * h)
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        col = (0, 255, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        name = coco[cid] if cid < len(coco) else str(cid)
        cv2.putText(out, '{} {:.2f}'.format(name, conf), (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
else:
    pass

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax[0].set_title('Input')
ax[0].axis('off')
ax[1].imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
ax[1].set_title('Output')
ax[1].axis('off')
plt.tight_layout()
plt.show()
