"""
Implements an analyser of the local similarity between two images, given the pre-computed matching regions.
"""

import cv2
import numpy
from skimage.exposure import match_histograms


# TODO add description.
def _compute_mutual_information(image1, image2, bins=256):
    # images must have the same shape
    image1 = image1.flatten()
    image2 = image2.flatten()

    # computes the ranges of pixel values within each given image
    min1 = numpy.min(image1)
    max1 = numpy.max(image1)
    step1 = 0.5 * (max1 - min1) / float(bins - 1)
    range1 = (min1 - step1, max1 + step1)

    min2 = numpy.min(image2)
    max2 = numpy.max(image2)
    step2 = 0.5 * (max2 - min2) / float(bins - 1)
    range2 = (min2 - step2, max2 + step2)

    # if useful ranges could be computed
    if range1[1] - range1[0] > 1.0 and range2[1] - range2[0] > 1.0:
        # computes the needed data histograms
        hist1, _ = numpy.histogram(image1, bins=bins, range=range1)
        hist2, _ = numpy.histogram(image2, bins=bins, range=range2)
        hist12, _, _ = numpy.histogram2d(image1, image2, bins=bins, range=[range1, range2])

        # computes the needed entropy (H1, H2, and H12)
        hist1 = hist1 / numpy.sum(hist1)
        hist1 = hist1[numpy.nonzero(hist1)]
        h1 = -1.0 * numpy.sum(hist1 * numpy.log2(hist1))

        hist2 = hist2 / numpy.sum(hist2)
        hist2 = hist2[numpy.nonzero(hist2)]
        h2 = -1.0 * numpy.sum(hist2 * numpy.log2(hist2))

        hist12 = hist12 / numpy.sum(hist12)
        hist12 = hist12[numpy.nonzero(hist12)]
        h12 = -1.0 * numpy.sum(hist12 * numpy.log2(hist12))

        # returns the mutual information
        return h1 + h2 - h12

    else:
        return 0.0


# TODO add description.
def compare(image1, keypoints1, image2, keypoints2, match_clusters):
    # holds the match-cluster-wise image-content mutual information after transforming image1 towards image2,
    # and vice-versa
    i1toi2_mi = []
    i2toi1_mi = []

    # holds all the matches previously computed between image1 and image2
    all_matches = []

    # for each cluster of matches...
    for c in match_clusters:
        for m in c:
            all_matches.append(m)

        # matched keypoints of the current cluster
        points1 = numpy.array([keypoints1[m.queryIdx].pt for m in c])
        points2 = numpy.array([keypoints2[m.trainIdx].pt for m in c])

        # transforms image 1 towards image 2 (warp, crop, color matching)
        homography1, _ = cv2.findHomography(points1, points2, cv2.LMEDS)
        warp_image1 = cv2.warpPerspective(image1, homography1, (image2.shape[1], image2.shape[0]),
                                          borderValue=(255, 255, 255))
        x2, y2, w2, h2 = cv2.boundingRect(points2.astype(numpy.int32))
        warp_image1 = warp_image1[y2:y2 + h2, x2:x2 + w2]
        crop_image2 = image2[y2:y2 + h2, x2:x2 + w2]
        warp_image1 = match_histograms(warp_image1, crop_image2, multichannel=True)

        # computes the current mutual information
        i1toi2_mi.append(_compute_mutual_information(warp_image1, crop_image2))

        # transforms image 2 towards image 1 (warp, crop, color matching)
        homography2, _ = cv2.findHomography(points2, points1, cv2.LMEDS)
        warp_image2 = cv2.warpPerspective(image2, homography2, (image1.shape[1], image1.shape[0]),
                                          borderValue=(255, 255, 255))
        x1, y1, w1, h1 = cv2.boundingRect(points1.astype(numpy.int32))
        crop_image1 = image1[y1:y1 + h1, x1:x1 + w1]
        warp_image2 = warp_image2[y1:y1 + h1, x1:x1 + w1]
        warp_image2 = match_histograms(warp_image2, crop_image1, multichannel=True)

        # computes the current mutual information
        i2toi1_mi.append(_compute_mutual_information(crop_image1, warp_image2))

    # returns the total number of matches, and mutual information
    return len(all_matches), i1toi2_mi, i2toi1_mi
