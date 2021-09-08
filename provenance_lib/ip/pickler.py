# TODO add description.

import numpy
import cv2


# TODO add description.
def pickle_keypoints(keypoints):
    output = []

    for image in keypoints:
        output1 = []
        if len(image) > 0 and len(image[0]) > 0:
            output1 = numpy.array([
                [image[0][i].pt[0], image[0][i].pt[1], image[0][i].size, image[0][i].angle, image[0][i].response]
                for i in range(len(image[0]))])

        output2 = []
        if len(image) > 1 and len(image[1]) > 0:
            output2 = numpy.array([
                [image[1][i].pt[0], image[1][i].pt[1], image[1][i].size, image[1][i].angle, image[1][i].response]
                for i in range(len(image[1]))])

        output.append([output1, output2])

    return output


# TODO add description.
def unpickle_keypoints(keypoint_arrays):
    output = []

    for image in keypoint_arrays:
        output1 = []
        if len(image[0]) > 0:
            output1 = [cv2.KeyPoint(image[0][i][0], image[0][i][1], image[0][i][2], image[0][i][3], image[0][i][4])
                       for i in range(len(image[0]))]

        output2 = []
        if len(image[1]) > 0:
            output2 = [cv2.KeyPoint(image[1][i][0], image[1][i][1], image[1][i][2], image[1][i][3], image[1][i][4])
                       for i in range(len(image[1]))]

        output.append([output1, output2])

    return output


# TODO add description.
def pickle_matches(matches):
    output = []

    for m_list in matches:
        if len(m_list) > 0:
            output.append(numpy.array([[m.queryIdx, m.trainIdx, m.distance] for m in m_list]))
        else:
            output.append([])

    return output


# TODO add description.
def unpickle_matches(match_array):
    output = []

    for m_list in match_array:
        if len(m_list) > 0:
            output.append([cv2.DMatch(int(ma[0]), int(ma[1]), ma[2]) for ma in m_list])
        else:
            output.append([])

    return output


# TODO add description.
def pickle_all_matches(matches):
    output = []

    for i_list in matches:
        output.append([])

        for m_list in i_list:
            if len(m_list) > 0:
                output[-1].append(numpy.array([[m.queryIdx, m.trainIdx, m.distance] for m in m_list]))
            else:
                output[-1].append([])

    return output


# TODO add description.
def unpickle_all_matches(match_array):
    output = []

    for i_list in match_array:
        output.append([])

        for m_list in i_list:
            if len(m_list) > 0:
                output[-1].append([cv2.DMatch(int(ma[0]), int(ma[1]), ma[2]) for ma in m_list])
            else:
                output[-1].append([])

    return output
