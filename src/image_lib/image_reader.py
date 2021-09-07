"""
Implements a reader of image files.
"""

import numpy
import cv2
import rawpy
import scipy
import PIL


# Reads the image stored in the given file path as a 3-channel numpy matrix.
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
            image = scipy.misc.imread(image_file_path, flatten=False)
        except:
            read_fail = True

    # tries to read the image with rawpy
    if read_fail:
        read_fail = False
        try:
            image = rawpy.imread(image_file_path).postprocess()
        except:
            read_fail = True

    # tries to read the image with plain pillow
    if read_fail:
        read_fail = False
        try:
            image = PIL.Image.open(image_file_path)
        except:
            read_fail = True

    # keep on trying with other libraries...

    # returns the obtained image
    if read_fail or image is None:
        print('[WARNING] Failed to read', image_file_path + '.')
        return None

    else:
        image = image.astype(numpy.uint8)
        channel_count = len(image.shape)

        if channel_count == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif channel_count > 3:
            image = image[:, :, 0:3]

        return image
