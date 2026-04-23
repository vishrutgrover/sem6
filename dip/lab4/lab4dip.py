from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open("/Users/vishrutgrover/coding/sem6/dip/ex3.png")
img = np.array(img)

def sampling(image, factor):
    h, w, c = image.shape
    sampled = image[::factor, ::factor]
    sampled_pil = Image.fromarray(sampled)
    sampled = np.array(sampled_pil.resize((w, h), Image.NEAREST))
    return sampled

def quantisation(image, levels):
    step = 256 // levels
    quantized = (image // step) * step
    return quantized

sample = sampling(img, 10)
quant = quantisation(img, 2)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(img)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("After Sampling")
plt.imshow(sample)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("After Quantisation")
plt.imshow(quant)
plt.axis("off")

plt.tight_layout()
plt.show()