import os

import numpy as np
from natsort import natsorted
import cv2
from PIL import Image

def data_loader(folder_dir):

    image_dataset = []

    image_files = natsorted(os.listdir(folder_dir))

    for image_name in image_files:
        image = cv2.imread(os.path.join(folder_dir, image_name), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (128, 128))
        image = Image.fromarray(image)
        image = np.array(image)
        image_dataset.append(image)

    return np.array(image_dataset)

image_dataset = data_loader("./data/images")
print(image_dataset)
mask_dataset = data_loader("./data/masks_machine")
print(mask_dataset)
