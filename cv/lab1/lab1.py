from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open("/Users/vishrutgrover/coding/sem6/cv/ex3.png")

plt.imshow(img)
plt.axis('off')
plt.show()

w, h = img.size
pixels = w*h
channels = len(img.getbands())
print(f"Size: {w}x{h}")
print(f"Pixels: {pixels}")
print(f"Channels: {channels}")

gray = img.convert('L')
plt.imshow(gray, cmap='gray')
plt.axis('off')
plt.show()

rgbim = img.convert('RGB')
imgarr = np.array(rgbim)

r = np.zeros_like(imgarr)
r[:, :, 0] = imgarr[:, :, 0]
rimg = Image.fromarray(r)

g = np.zeros_like(imgarr)
g[:, :, 1] = imgarr[:, :, 1]
gimg = Image.fromarray(g)

b = np.zeros_like(imgarr)
b[:, :, 2] = imgarr[:, :, 2]
bimg = Image.fromarray(b)

plt.imshow(rimg)
plt.axis('off')
plt.show()

plt.imshow(gimg)
plt.axis('off')
plt.show()

plt.imshow(bimg)
plt.axis('off')
plt.show()

scaled = img.resize((img.size[0]//2, img.size[1]//2))
plt.imshow(scaled)
plt.axis('off')
plt.show()

rot = img.rotate(90)
plt.imshow(rot)
plt.axis('off')
plt.show()

trans = img.transform(img.size, Image.AFFINE, (1, 0, 500, 0, 1, 500))
plt.imshow(trans)
plt.axis('off')
plt.show()