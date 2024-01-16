import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import random
from PIL import Image

random.seed(55)

image_path = Path("data/pizza_steak_sushi")
image_path_list = list(image_path.glob("*/*/*.jpg"))
print(len(image_path_list), image_path_list[0])

random_img_path = random.choice(image_path_list)
print(random_img_path, type(random_img_path))

image_class = random_img_path.parent.stem
print(f'class name: {image_class}')

img = Image.open(random_img_path)
print(f'img shape: {img.height} * {img.width}')

img_as_array = np.asarray(img)
# Plot the image with matplotlib
plt.figure(figsize=(10, 7))
plt.imshow(img_as_array)
plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape} -> [height, width, color_channels]")
plt.axis(False)
plt.show()