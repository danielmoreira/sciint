"""
Implements a descriptor of image files.
"""

import cv2
import imutils
import numpy
import pytesseract

import provenance_lib.ip.pickler as pickler

CLAHE_APPLIER = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# Keeps doubling the size of the given <image> and respective <mask> up to the point of having
# more pixels than <min_image_size>.
# Returns the re-sized image and mask, and the number of times they were re-sized.
def _increase_image_if_necessary(image, mask=None, min_image_size=10000):
    resize_count = 0

    while image.shape[0] * image.shape[1] < min_image_size:
        image = cv2.resize(image, (0, 0), fx=2.0, fy=2.0)

        if mask is not None:
            mask = cv2.resize(mask, (0, 0), fx=2.0, fy=2.0)

        resize_count = resize_count + 1

    return image, mask, resize_count


# Masks out all the text within a given gray-scaled image <gs_image>.
# Returns an image mask whose white pixels represent the regions without text.
def _mask_text_out(gs_image):
    # original input image dimensions
    oh, ow = gs_image.shape

    # image size perturbation, useful to improve tesseract
    gs_image = cv2.resize(gs_image, None, fx=1.2, fy=1.2)

    # image rotated versions
    gs_images = [gs_image,
                 imutils.rotate_bound(gs_image, -90), imutils.rotate_bound(gs_image, -45),
                 imutils.rotate_bound(gs_image, +45), imutils.rotate_bound(gs_image, +90)]

    # for each rotated version, obtains the respective text mask
    masks = []
    for gs_image in gs_images:
        # rotated image dimensions
        h, w = gs_image.shape

        # current mask
        mask = numpy.ones((h, w), numpy.uint8) * 255

        # detects and masks out all the text within the current image
        tesse_data = pytesseract.image_to_data(gs_image).split('\n')[1:]  # 0th row is header
        for row in tesse_data:
            items = row.split()
            if len(items) == 12 and float(items[10]) > 75:  # 75 is the confidence (between 0 and 100)
                has_alpha = False
                for c in str(items[11]):
                    if c.isalnum():
                        has_alpha = True
                        break

                if has_alpha:
                    r0 = int(items[6])
                    c0 = int(items[7])
                    r1 = r0 + int(items[8])
                    c1 = c0 + int(items[9])

                    cv2.rectangle(mask, (r0, c0), (r1, c1), (0, 0, 0), -1)

        masks.append(mask)

    # properly de-rotates the obtained masks
    h, w = masks[0].shape

    masks[1] = imutils.rotate_bound(masks[1], +90)

    masks[2] = imutils.rotate_bound(masks[2], +45)
    h2 = int((masks[2].shape[0] - h) / 2)
    w2 = int((masks[2].shape[1] - w) / 2)
    masks[2] = masks[2][h2:h2 + h, w2:w2 + w]

    masks[3] = imutils.rotate_bound(masks[3], -45)
    h3 = int((masks[3].shape[0] - h) / 2)
    w3 = int((masks[3].shape[1] - w) / 2)
    masks[3] = masks[3][h3:h3 + h, w3:w3 + w]

    masks[4] = imutils.rotate_bound(masks[4], -90)

    # combines the obtained masks
    mask = masks[0]
    mask = cv2.bitwise_and(mask, mask, mask=masks[1])
    mask = cv2.bitwise_and(mask, mask, mask=masks[2])
    mask = cv2.bitwise_and(mask, mask, mask=masks[3])
    mask = cv2.bitwise_and(mask, mask, mask=masks[4])

    # returns the computed mask
    mask = cv2.resize(mask, (ow, oh))
    return mask


# Detects SIFT keypoints over the given <image> and describes them with RootSIFT.
# It extracts up to <kp_count> samples, obeying the eventually given image <mask>.
# Give <mask_text> as True if you want any overlay text on the image to be ignored during keypoint detection.
# Give <mask_background> as True if want to ignore an eventual white background during keypoint detection.
# Returns the obtained keypoints with their respective descriptions, as well as the resulting keypoint detection mask.
def sift_detect_rsift_describe(image, kp_count=2000, mask=None, mask_text=True, mask_background=True, eps=1e-7):
    image, mask, resize_count = _increase_image_if_necessary(image, mask)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # masks text out, if it is the case
    if mask_text:
        text_mask = _mask_text_out(image)
        if mask is None:
            mask = text_mask
        else:
            mask = cv2.bitwise_and(text_mask, text_mask, mask=mask)

    # masks white background out
    if mask_background:
        _, bg_mask = cv2.threshold(image, 254, 255, cv2.THRESH_BINARY_INV)
        if mask is None:
            mask = bg_mask
        else:
            mask = cv2.bitwise_and(bg_mask, bg_mask, mask=mask)

    # applies CLAHE histogram equalization over the given image
    image = CLAHE_APPLIER.apply(image)

    # SIFT detector and descriptor
    sift_detector = cv2.xfeatures2d.SIFT_create(nfeatures=kp_count, contrastThreshold=0.0, sigma=3.2)

    # detects SIFT keypoints over the given image
    keypoints = sift_detector.detect(image, mask=mask)

    # if no keypoints were detected, returns empty keypoints and descriptions
    if len(keypoints) == 0:
        return [], [], None

    # else...
    # sorts the obtained keypoints according to their response, and keeps only the top-<kp_count> ones
    keypoints.sort(key=lambda k: k.response, reverse=True)
    del keypoints[kp_count:]

    # describes the remaining keypoints
    keypoints, descriptions = sift_detector.compute(image, keypoints)
    descriptions = descriptions / (descriptions.sum(axis=1, keepdims=True) + eps)
    descriptions = numpy.sqrt(descriptions)

    # re-adjusts the obtained keypoints according to the change of image size
    if resize_count > 0:
        for kp in keypoints:
            kp.pt = (kp.pt[0] / (2.0 ** resize_count), kp.pt[1] / (2.0 ** resize_count))
            kp.size = kp.size / (2.0 ** resize_count)

        if mask is not None:
            for i in range(0, resize_count):
                mask = cv2.resize(mask, (0, 0), fx=0.5, fy=0.5)

    # returns the obtained keypoints and their respective descriptions
    return keypoints, descriptions, mask


# TODO add description.
def describe(image, kp_count=2000):
    # describes the regular content of the given image
    regl_keypoints, regl_descriptions, mask = sift_detect_rsift_describe(image, kp_count=kp_count)

    # describes the flipped content of the given image
    flip_keypoints, flip_descriptions, _ = sift_detect_rsift_describe(cv2.flip(image, 1), kp_count=kp_count,
                                                                      mask=cv2.flip(mask, 1), mask_text=False,
                                                                      mask_background=False)

    # returns the obtained data
    return pickler.pickle_keypoints([[regl_keypoints, flip_keypoints]]), (regl_descriptions, flip_descriptions)
