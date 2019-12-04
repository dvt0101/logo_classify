import numpy as np
import imgaug as ia
import cv2
import os
import imgaug.augmenters as iaa
from PIL import Image
ia.seed(1)

path = '/home/vietthangtik15/cv/classify_logo'
folder_img = os.listdir(path)
print(folder_img)
images = []
for folder in folder_img:
    path_f = os.path.join(path, folder)
    images = [cv2.imread(os.path.join(path_f, img)) for img in os.listdir(path_f)]


seq1 = iaa.Sequential([
    iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 3.0)),
    iaa.ContrastNormalization((0.75, 1.5)),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    ) 
])


images_aug = seq(images=images[0])
print(images_aug.shape)
cv2.imwrite('before.jpg', images[0])
cv2.imwrite('augm.png', images_aug)