import cv2
import numpy as np

path = '/Users/vishrutgrover/coding/sem6/dip/lab6/'
img = cv2.imread(path + 'objects.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

for i in range(8):
    plane = (gray >> i) & 1
    cv2.imwrite(path + f'bit{i}.png', plane * 255)