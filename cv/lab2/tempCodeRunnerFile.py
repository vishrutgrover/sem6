from PIL import Image
img = Image.open('/Users/vishrutgrover/coding/sem6/cv/lab2/claw.png')
left, top, right, bottom = 300, 100, 600, 400
cropped = img.crop((left, top, right, bottom))
cropped.save('/Users/vishrutgrover/coding/sem6/cv/lab2/cropclaw.png')
print("Image cropped and saved as 'cropclaw.png'")