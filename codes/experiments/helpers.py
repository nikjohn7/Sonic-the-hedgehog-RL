import numpy as np
import skimage
from skimage.viewer import ImageViewer


def resize_img(observation, size):
    img_gray = skimage.color.rgb2gray(observation)
    resized = skimage.transform.resize(img_gray, size)
    return resized

def view_img(img):
    viewer = ImageViewer(img)
    viewer.show()