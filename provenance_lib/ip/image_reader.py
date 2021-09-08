"""
Implements a reader of image files.
"""

import os
import random
import numpy
import cv2
import rawpy
import imageio
import scipy
import PIL


# Reads the image stored in the given file path as a 3-channel BGR numpy matrix.
def read(image_file_path):
    image = None
    read_fail = True

    # tries to read the image with opencv
    if read_fail:
        read_fail = False
        try:
            image = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
        except:
            read_fail = True

    # tries to read the image with scipy
    if read_fail:
        read_fail = False
        try:
            image = cv2.cvtColor(scipy.misc.imread(image_file_path, flatten=False, mode='RGB'), cv2.COLOR_RGB2BGR)

        except:
            read_fail = True

    # tries to read the image with rawpy
    if read_fail:
        read_fail = False
        try:
            image = rawpy.imread(image_file_path).postprocess()

            random_file_path = str(random.randint(0, 1000000)) + str(random.randint(0, 1000000)) + '.png'
            imageio.imsave(random_file_path, image)

            image = cv2.imread(random_file_path, cv2.IMREAD_COLOR)
            os.remove(random_file_path)

        except:
            read_fail = True

    # tries to read the image with plain pillow
    if read_fail:
        read_fail = False
        try:
            image = cv2.cvtColor(numpy.array(PIL.Image.open(image_file_path)), cv2.COLOR_RGB2BGR)
        except:
            read_fail = True

    # keep on trying with other libraries...

    # returns the obtained image
    if read_fail or image is None:
        print('[WARNING] Failed to read', image_file_path + '.')
        return numpy.zeros((64, 64, 3), numpy.uint8)

    else:
        image = image.astype(numpy.uint8)
        shape_dims = len(image.shape)

        if shape_dims == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif shape_dims > 3:
            image = image[:, :, 0:3]

        return image
