# WAP to implement transformation of an image (scaling, rotation, resizing)
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("/Users/vishrutgrover/coding/sem6/dip/ex3.png")

scaled = img.resize((img.size[0]//2, img.size[1]//2))
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(scaled)
plt.title('Scaled (50%)')
plt.axis('off')

rotated = img.rotate(90)
plt.subplot(1, 3, 2)
plt.imshow(rotated)
plt.title('Rotated (90°)')
plt.axis('off')

resized = img.resize((1000, 500))
plt.subplot(1, 3, 3)
plt.imshow(resized)
plt.title('Resized (1000x500)')
plt.axis('off')

plt.tight_layout()
plt.show()