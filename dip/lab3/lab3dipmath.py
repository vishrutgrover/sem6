import numpy as np
import math
from PIL import Image

img = Image.open("/Users/vishrutgrover/coding/sem6/dip/ex3.png")
imgarr = np.array(img)
h, w = imgarr.shape[:2]

def scale(img, factor):
    nh, nw = int(h * factor), int(w * factor)
    scaled = np.zeros((nh, nw, img.shape[2]), dtype=img.dtype)
    for i in range(nh):
        for j in range(nw):
            si, sj = int(i / factor), int(j / factor)
            scaled[i, j] = img[si, sj]
    return scaled

def rotate(img, angle):
    rad = math.radians(angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    nh, nw = h, w
    rotated = np.zeros((nh, nw, img.shape[2]), dtype=img.dtype)
    for i in range(nh):
        for j in range(nw):
            x, y = j - cx, i - cy
            xr = int(x * cos_a - y * sin_a + cx)
            yr = int(x * sin_a + y * cos_a + cy)
            if 0 <= yr < h and 0 <= xr < w:
                rotated[i, j] = img[yr, xr]
    return rotated

def resize(img, new_h, new_w):
    h, w = img.shape[:2]
    resized = np.zeros((new_h, new_w, img.shape[2]), dtype=img.dtype)
    for i in range(new_h):
        for j in range(new_w):
            si, sj = int(i * h / new_h), int(j * w / new_w)
            resized[i, j] = img[si, sj]
    return resized

scaled = scale(imgarr, 0.5)
rotated = rotate(imgarr, 120)
resized = resize(imgarr, 10, 50)
Image.fromarray(scaled).save("/Users/vishrutgrover/coding/sem6/dip/lab3/scaled.png")
Image.fromarray(rotated).save("/Users/vishrutgrover/coding/sem6/dip/lab3/rotated.png")
Image.fromarray(resized).save("/Users/vishrutgrover/coding/sem6/dip/lab3/resized.png")
print("Transformations saved: scaled.png, rotated.png, resized.png")