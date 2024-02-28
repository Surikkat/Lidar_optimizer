import os

import numpy as np
import pandas as pd
from natsort import natsorted
import cv2
from PIL import Image

def rgb_to_labels(img, mask_labels):

    label_seg = np.zeros(img.shape,dtype=np.uint8)

    for i in range(mask_labels.shape[0]):
        label_seg[np.all(img == list(mask_labels.iloc[i, [1,2,3]]), axis=-1)] = i

    label_seg = label_seg[:,:,0]

    return label_seg

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
mask_dataset = data_loader("./data/masks_machine")
mask_labels = pd.read_csv('./data/class_dict.csv')

labels = []
for i in range(mask_dataset.shape[0]):
    label = rgb_to_labels(mask_dataset[i], mask_labels)
    labels.append(label)

labels = np.array(labels)
labels = np.expand_dims(labels, axis=3)
print(labels)

