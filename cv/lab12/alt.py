import cv2
import numpy as np

# Load class names (COCO)
classes = open("coco.names").read().strip().split("\n")

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load image
img = cv2.imread("image.jpg")
h, w = img.shape[:2]

# Create blob
blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Get output layers
layer_names = net.getUnconnectedOutLayersNames()
outputs = net.forward(layer_names)

# Loop over detections
for output in outputs:
    for det in output:
        scores = det[5:]
        class_id = np.argmax(scores)
        conf = scores[class_id]

        if conf > 0.5:
            cx, cy, bw, bh = (det[0:4] * np.array([w, h, w, h])).astype(int)
            x = int(cx - bw/2)
            y = int(cy - bh/2)

            cv2.rectangle(img, (x, y), (x+bw, y+bh), (0,255,0), 2)
            cv2.putText(img, classes[class_id], (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

# Show result
cv2.imshow("Output", img)
cv2.waitKey(0)
cv2.destroyAllWindows()