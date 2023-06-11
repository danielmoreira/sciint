"""
Implements a matcher of keypoints.
"""

import numpy
import cv2
import math


# Selects, from the given set of <matches>, the largest set of matches that are geometrically consistent
# with the <i>-th and <j>-th given matches. The matches are of the OpenCV DMatch type and therefore need
# the sets of keypoints <keypoints1> and <keypoints2> to compute consistency. The displacement threshold
# <displacement_thresh> is a tolerance, in pixels, to consider deviant matches still consistent.
# Returns the indices of the selected matches.
def __select_consistent_matches(matches, i, j, keypoints1, keypoints2, displacement_thresh=15):
    # if the given ij matches share keypoints, there is nothing to select
    if matches[i].queryIdx == matches[j].queryIdx or matches[i].trainIdx == matches[j].trainIdx:
        return []

    # query and train keypoints in matrix format
    query_points = numpy.array([(kp.pt[0], kp.pt[1]) for kp in keypoints1])
    train_points = numpy.array([(kp.pt[0], kp.pt[1]) for kp in keypoints2])

    # puts query and train images in the same scale
    query_i_point = query_points[matches[i].queryIdx]
    query_j_point = query_points[matches[j].queryIdx]
    query_ij_distance = math.sqrt(
        (query_i_point[0] - query_j_point[0]) ** 2 + (query_i_point[1] - query_j_point[1]) ** 2)

    train_i_point = train_points[matches[i].trainIdx]
    train_j_point = train_points[matches[j].trainIdx]
    train_ij_distance = math.sqrt(
        (train_i_point[0] - train_j_point[0]) ** 2 + (train_i_point[1] - train_j_point[1]) ** 2)

    # if the involved keypoints are too close to eah other, there is nothing to select
    if query_ij_distance == 0.0 or train_ij_distance == 0.0:
        return []

    distance_ratio = query_ij_distance / train_ij_distance

    if distance_ratio > 1.0:
        scale_matrix = numpy.zeros((3, 3))
        scale_matrix[0, 0] = distance_ratio
        scale_matrix[1, 1] = distance_ratio
        scale_matrix[2, 2] = 1.0
        train_points = cv2.perspectiveTransform(numpy.float32([train_points]), scale_matrix)[0]

    elif distance_ratio < 1.0:
        scale_matrix = numpy.zeros((3, 3))
        scale_matrix[0, 0] = 1.0 / distance_ratio
        scale_matrix[1, 1] = 1.0 / distance_ratio
        scale_matrix[2, 2] = 1.0
        query_points = cv2.perspectiveTransform(numpy.float32([query_points]), scale_matrix)[0]

    # computes and performs the rotation of the train image towards the query image
    query_i_point = query_points[matches[i].queryIdx]
    query_j_point = query_points[matches[j].queryIdx]
    query_angle = math.atan2(query_j_point[1] - query_i_point[1], query_j_point[0] - query_i_point[0])
    if query_angle < 0.0:
        query_angle = 2.0 * math.pi + query_angle

    train_i_point = train_points[matches[i].trainIdx]
    train_j_point = train_points[matches[j].trainIdx]
    train_angle = math.atan2(train_j_point[1] - train_i_point[1], train_j_point[0] - train_i_point[0])
    if train_angle < 0.0:
        train_angle = 2.0 * math.pi + train_angle

    train_angle_correction = query_angle - train_angle
    sine = math.sin(train_angle_correction)
    cosine = math.cos(train_angle_correction)

    rotation_matrix = numpy.zeros((3, 3))
    rotation_matrix[0, 0] = cosine
    rotation_matrix[0, 1] = -sine
    rotation_matrix[1, 0] = sine
    rotation_matrix[1, 1] = cosine
    rotation_matrix[2, 2] = 1.0

    train_points = cv2.perspectiveTransform(numpy.float32([train_points]), rotation_matrix)[0]

    # computes and performs the translation of the train image towards the query image
    query_i_point = query_points[matches[i].queryIdx]
    train_i_point = train_points[matches[i].trainIdx]

    translation_matrix = numpy.zeros((3, 3))
    translation_matrix[0, 0] = 1.0
    translation_matrix[1, 1] = 1.0
    translation_matrix[2, 2] = 1.0
    translation_matrix[0, 2] = query_i_point[0] - train_i_point[0]
    translation_matrix[1, 2] = query_i_point[1] - train_i_point[1]

    train_points = cv2.perspectiveTransform(numpy.float32([train_points]), translation_matrix)[0]

    # selects the geometrically consistent matches
    selected_matches = []

    used_query_idx = []
    used_train_idx = []

    used_query_pt = []
    used_train_pt = []

    for m in range(len(matches)):
        if matches[m].queryIdx in used_query_idx or matches[m].trainIdx in used_train_idx:
            continue

        query_x = query_points[matches[m].queryIdx][0]
        query_y = query_points[matches[m].queryIdx][1]

        train_x = train_points[matches[m].trainIdx][0]
        train_y = train_points[matches[m].trainIdx][1]

        if (query_x, query_y) in used_query_pt or (train_x, train_y) in used_train_pt:
            continue

        if math.sqrt((query_x - train_x) ** 2 + (query_y - train_y) ** 2) < displacement_thresh:
            selected_matches.append(m)

            used_query_idx.append(matches[m].queryIdx)
            used_train_idx.append(matches[m].trainIdx)

            used_query_pt.append((query_x, query_y))
            used_train_pt.append((train_x, train_y))

    # returns the selected consistent matches
    return selected_matches


# Performs G2NN selection over the given two sets of keypoints and their respective descriptions
# (<keypoints1>, <descriptions1>), (<keypoints2>, <descriptions2>). Parameter <k_rate>
# in [0.0, 1.0] helps to define how many neighbors are matched to each given keypoint. Parameter
# <nndr_threshold> in [0.0, 1.0] is the maximum value used to consider a match useful, according to its
# difference (distance-wise) to the next closest match (G2NN principle).
# Returns the indices of the selected keypoints, for each one of the given two sets.
def _g2nn_keypoint_selection(keypoints1, descriptions1, keypoints2, descriptions2,
                             k_rate=0.5, nndr_threshold=0.75, eps=1e-7):
    # defines the two sets of keypoints to be matched
    # (smaller set: keypoints1, larger set: keypoints2)
    swapped = False
    if len(keypoints2) < len(keypoints1) or (
            len(keypoints2) == len(keypoints1) and keypoints2[0].size < keypoints1[0].size):
        keypoints1, keypoints2 = keypoints2, keypoints1
        descriptions1, descriptions2 = descriptions2, descriptions1
        swapped = True

    # matches keypoints1 towards keypoints2
    knn_matches = cv2.BFMatcher().knnMatch(descriptions1, descriptions2, k=max(1, int(round(len(keypoints1) * k_rate))))

    # g2NN match selection
    selected_matches = []
    for _, matches in enumerate(knn_matches):
        for i in range(0, len(matches) - 1):
            if matches[i].distance / (matches[i + 1].distance + eps) < nndr_threshold:
                selected_matches.append(matches[i])
            else:
                break

    # keypoint and description selection, based on selected matches
    indices1 = []
    indices2 = []
    for m in selected_matches:
        if m.queryIdx not in indices1:
            indices1.append(m.queryIdx)

        if m.trainIdx not in indices2:
            indices2.append(m.trainIdx)

    selected_keypoints_1 = [index for index in indices1]
    selected_keypoints_2 = [index for index in indices2]

    # undoes swapping, if it is the case
    if swapped:
        selected_keypoints_1, selected_keypoints_2 = selected_keypoints_2, selected_keypoints_1

    # returns the selected keypoints
    return selected_keypoints_1, selected_keypoints_2


# Computes, from the given two sets of keypoints and their respective descriptions (<keypoints1>, <descriptions1>),
# (<keypoints2>, <descriptions2>), the largest set of matches that are geometrically consistent among themselves.
# Parameter <nndr_thresh> in [0.0, 1.0] helps to select useful keypoints (2NN, a.k.a. Lowe's NNDR selection).
# Parameter <match_proc_upb> sets the upper bound (number of keypoint matches) to stop processing matches,
# for faster non-exhaustive runtime. Parameter <selected_match_upb> sets the upper bound (in percentage of
# selected good matches) to stop processing matches for faster non-exhaustive runtime.
# Returns a list of matches of the OpenCV DMatch type.
def _consistent_match(keypoints1, descriptions1, keypoints2, descriptions2,
                      nndr_thresh=0.85, match_proc_upb=100, selected_match_upb=0.5, eps=1e-7, ):
    # defines the two sets of keypoints to be matched
    # (smaller set: keypoints1, larger set: keypoints2)
    swapped = False
    if len(keypoints2) < len(keypoints1) or (
            len(keypoints2) == len(keypoints1) and keypoints2[0].size < keypoints1[0].size):
        keypoints1, keypoints2 = keypoints2, keypoints1
        descriptions1, descriptions2 = descriptions2, descriptions1
        swapped = True

    # matches keypoints1 towards keypoints2 and performs 2NN (NNDR) verification
    knn_matches = cv2.BFMatcher().knnMatch(descriptions1, descriptions2, k=2)
    good_matches = []
    for _, matches in enumerate(knn_matches):
        if len(matches) > 1 and matches[0].distance / (matches[1].distance + eps) < nndr_thresh:
            good_matches.append(matches[0])
    good_matches.sort(key=lambda m: m.distance)

    # selects the largest set of geometrically consistent matches
    selected_matches = []
    for i in range(0, len(good_matches) - 1):
        # if enough consistent matches were found, time do quit loop
        if len(good_matches) > match_proc_upb and len(selected_matches) >= selected_match_upb * len(good_matches):
            break

        for j in range(i + 1, len(good_matches)):
            # if enough consistent matches were found, time do quit loop
            if len(good_matches) > match_proc_upb and len(selected_matches) >= selected_match_upb * len(good_matches):
                break

            consistent_matches = __select_consistent_matches(good_matches, i, j, keypoints1, keypoints2)
            if len(consistent_matches) > len(selected_matches):
                selected_matches = consistent_matches[:]

    # prepares the selected matches to be returned
    answer = []
    for i in selected_matches:
        # fixes swap, if needed
        if swapped:
            good_matches[i].queryIdx, good_matches[i].trainIdx = good_matches[i].trainIdx, good_matches[i].queryIdx

        answer.append(good_matches[i])

    answer.sort(key=lambda m: m.distance)
    return answer


# Matches the content of image1 (<image_shape21>, <keypoints1>, <descriptions1>) with the content of image2
# (<image_shape2>, <keypoints2>, <descriptions2>), keeping geometric consistency (e.g., no matches crossing
# each other). Parameter <min_match_count> defines the minimum number of matches to consider a set valid.
# Returns a list of OpenCV DMatch objects.
def match(keypoints1, descriptions1, keypoints2, descriptions2,
          min_match_count=9):
    if len(keypoints1) == 0 or len(keypoints2) == 0:
        return []

    # defines the two sets of keypoints to be matched
    # (smaller set: keypoints1, larger set: keypoints2)
    swapped = False
    if len(keypoints2) < len(keypoints1) or (
            len(keypoints2) == len(keypoints1) and keypoints2[0].size < keypoints1[0].size):
        keypoints1, keypoints2 = keypoints2, keypoints1
        descriptions1, descriptions2 = descriptions2, descriptions1
        swapped = True

    # performs g2nn keypoint selection
    keypoint_indices1, keypoint_indices2 = _g2nn_keypoint_selection(keypoints1, descriptions1,
                                                                    keypoints2, descriptions2)
    if len(keypoint_indices1) == 0 or len(keypoint_indices2) == 0:
        return []

    selected_keypoints1 = [keypoints1[i] for i in keypoint_indices1]
    selected_descriptions1 = numpy.array([descriptions1[i] for i in keypoint_indices1], dtype=numpy.float32)
    selected_keypoints2 = [keypoints2[i] for i in keypoint_indices2]
    selected_descriptions2 = numpy.array([descriptions2[i] for i in keypoint_indices2], dtype=numpy.float32)

    # finds the geometrically consistent matches
    ij_matches = _consistent_match(selected_keypoints1, selected_descriptions1,
                                   selected_keypoints2, selected_descriptions2)
    if len(ij_matches) >= min_match_count:
        # updates matches indices
        for m in ij_matches:
            m.queryIdx = keypoint_indices1[m.queryIdx]
            m.trainIdx = keypoint_indices2[m.trainIdx]

            # fixes swap, if needed
            if swapped:
                m.queryIdx, m.trainIdx = m.trainIdx, m.queryIdx

        # returns the selected matches
        return [ij_matches]

    else:
        # otherwise, no matches were found
        return []
