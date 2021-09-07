"""
Implements a descriptor of image files.
"""

import cv2
import numpy


# Keeps doubling the given image size up to the point of having more pixels than <min_image_size>.
# The image aspect ratio is kept.
# Returns the resized image and the number of times it was resized.
def _increase_image_if_necessary(image, min_image_size):
    resize_count = 0

    while image.shape[0] * image.shape[1] < min_image_size:
        image = cv2.resize(image, (0, 0), fx=2, fy=2)
        resize_count = resize_count + 1

    return image, resize_count


# Detects SURF keypoints over the given image and describes them with RootSIFT.
# It extracts up to <kp_count> samples, obeying the eventually given image mask.
def surf_detect_rsift_describe(image, mask=None, kp_count=500, min_image_size=100000, hessian=1, eps=1e-7):
    image, resize_count = _increase_image_if_necessary(image, min_image_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # SURF detector
    surf_detector = cv2.xfeatures2d.SURF_create(hessian)

    # SIFT descriptor
    sift_detector = cv2.xfeatures2d.SIFT_create()

    # detects SURF keypoints over the given image
    keypoints = surf_detector.detect(image, mask)

    # if no keypoints were detected, creates a single one at the middle of the image
    if len(keypoints) == 0:
        x = int(image.shape[1] / 2.0)
        y = int(image.shape[0] / 2.0)
        size = min(min(image.shape), 64 * (resize_count + 1))
        keypoints.append(cv2.KeyPoint(x, y, size))

    # orders the obtained interest points according to their response, and keeps only the top-<kp_count> ones
    keypoints = sorted(keypoints, key=lambda k: k.response, reverse=True)
    del keypoints[kp_count:]

    # describes the remaining interest points
    keypoints, descriptions = sift_detector.compute(image, keypoints)
    descriptions = descriptions / (descriptions.sum(axis=1, keepdims=True) + eps)
    descriptions = numpy.sqrt(descriptions)

    # adjusts the obtained keypoints according to the change of image size
    if resize_count > 0:
        for kp in keypoints:
            kp.pt = (kp.pt[0] / (2.0 ** resize_count), kp.pt[1] / (2.0 ** resize_count))
            kp.size = kp.size / (2.0 ** resize_count)

    # returns the obtained keypoints and their respective descriptions
    return keypoints, descriptions


# Stores the given interest points and their respective descriptions into the given file path.
# Format: x y s a r d+
def store_descriptions(keypoints, descriptions, description_file_path):
    if len(keypoints) != len(descriptions):
        raise Exception('[ERROR] Number of keypoints and descriptions is not the same. k:',
                        str(len(keypoints)) + ', d:', str(len(descriptions)) + '.')

    content = []

    if len(keypoints) > 0:
        content = numpy.zeros((len(keypoints), descriptions.shape[1] + 5), numpy.float32)

        for i in range(len(keypoints)):
            content[i][0] = keypoints[i].pt[0]
            content[i][1] = keypoints[i].pt[1]
            content[i][2] = keypoints[i].size
            content[i][3] = keypoints[i].angle
            content[i][4] = keypoints[i].response
            content[i][5:] = descriptions[i][:]

    numpy.save(description_file_path, content)


# Loads the image descriptions stored in the given file path.
# Returns the loaded interest points and their respective descriptions.
def load_descriptions(description_file_path):
    keypoints = []
    descriptions = []

    content = numpy.load(description_file_path)
    for c in content:
        x = c[0]
        y = c[1]
        size = c[2]
        angle = c[3]
        response = c[4]
        description = c[5:].astype(numpy.float32)

        keypoints.append(cv2.KeyPoint(x, y, size, angle, response))
        if len(descriptions) == 0:
            descriptions.append(description)
            descriptions = numpy.array(descriptions)
        else:
            descriptions = numpy.vstack((descriptions, description))

    return keypoints, descriptions
