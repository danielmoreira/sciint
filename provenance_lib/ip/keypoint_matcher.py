"""
Implements a matcher of keypoints.
"""

import numpy
import cv2
import math


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph


# Selects, from the given set of <matches>, the largest set of matches that are geometrically consistent
# with the <i>-th and <j>-th given matches. The matches are of the OpenCV DMatch type and therefore need
# the sets of keypoints <keypoints1> and <keypoints2> to compute consistency. The displacement threshold
# <displacement_thresh> is a tolerance, in pixels, to consider deviant matches still consistent.
# Returns the indices of the selected matches.
def __select_consistent_matches(matches, i, j, keypoints1, keypoints2, displacement_thresh=15):
    # selected matches
    selected_matches = []

    # if the given ij matches share keypoints, there is nothing to select
    if matches[i].queryIdx == matches[j].queryIdx or matches[i].trainIdx == matches[j].trainIdx:
        return selected_matches

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
        return selected_matches

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
# (<keypoints1>, <descriptions1>), (<keypoints2>, <descriptions2>). Please provide <within_same_image>
# as TRUE in the case of the two keypoint sets coming from the same (single) image. Parameter <k_rate>
# in [0.0, 1.0] helps to define how many neighbors are matched to each given keypoint. Parameter
# <nndr_threshold> in [0.0, 1.0] is the maximum value used to consider a match useful, according to its
# difference (distance-wise) to the next closest match (G2NN principle).
# Returns the indices of the selected keypoints, for each one of the given two sets.
def _g2nn_keypoint_selection(keypoints1, descriptions1, keypoints2, descriptions2,
                             within_same_image=False, k_rate=0.5, nndr_threshold=0.75, eps=1e-7):
    # defines the two sets of keypoints to be matched
    # (smaller set: keypoints1, larger set: keypoints2)
    swapped = False
    if len(keypoints2) < len(keypoints1):
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
    for match in selected_matches:
        # if keypoint self comparison, then does nothing and goes to the next iteration
        if within_same_image and match.queryIdx == match.trainIdx:
            continue

        if match.queryIdx not in indices1:
            indices1.append(match.queryIdx)

        if match.trainIdx not in indices2:
            indices2.append(match.trainIdx)

    selected_keypoints_1 = [index for index in indices1]
    selected_keypoints_2 = [index for index in indices2]

    # undoes swapping, if it is the case
    if swapped:
        selected_keypoints_1, selected_keypoints_2 = selected_keypoints_2, selected_keypoints_1

    # returns the selected keypoints
    return selected_keypoints_1, selected_keypoints_2


# Clusterizes the given <keypoints> according to their (x, y) positions within the source image.
# The shape (height, width) <image_shape> of the source image must also be given.
# Additional clustering configuration parameters:
# <conn_neighbor_rate> - Rate in [0.0, 1.0] to define the number of keypoints used to lock connectivity in the
#                        clustering solution;
#   <dist_thresh_rate> - Rate in [0.0, 1.0] to define the distance threshold used in the clustering solution;
#          <cpu_count> - Number of CPU cores used in clustering. Give -1 to use all cores.
# Returns the computed clusters as a list of lists containing the indices of the clustered keypoints.
def _clusterize_keypoints(keypoints, image_shape, conn_neighbor_rate=0.1, dist_thresh_rate=0.003, cpu_count=-1):
    if len(keypoints) == 0:
        return [[]]

    if len(keypoints) == 1:
        return [[0]]

    positions = numpy.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])

    nb_count = int(round(len(positions) * conn_neighbor_rate))
    dist_thresh = image_shape[0] * image_shape[1] * dist_thresh_rate

    if nb_count > 0:
        forced_conn = kneighbors_graph(X=positions, n_neighbors=nb_count, n_jobs=cpu_count)
        clustering = AgglomerativeClustering(n_clusters=None, connectivity=forced_conn, distance_threshold=dist_thresh)
    else:
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=dist_thresh)

    clustering.fit(positions)
    labels = clustering.labels_

    clusters = {}
    for i in range(0, len(labels)):
        label = labels[i]

        if label not in clusters.keys():
            clusters[label] = []

        clusters[label].append(i)

    return [clusters[label] for label in clusters.keys()]


# Computes, from the given two sets of keypoints and their respective descriptions (<keypoints1>, <descriptions1>),
# (<keypoints2>, <descriptions2>), the largest set of matches that are geometrically consistent among themselves.
# Parameter <nndr_thresh> in [0.0, 1.0] helps to select useful keypoints (2NN, a.k.a. Lowe's NNDR selection).
# Returns a list of matches of the OpenCV DMatch type.
def _consistent_match(keypoints1, descriptions1, keypoints2, descriptions2, nndr_thresh=0.85, eps=1e-7):
    # defines the two sets of keypoints to be matched
    # (smaller set: keypoints1, larger set: keypoints2)
    swapped = False
    if len(keypoints2) < len(keypoints1):
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
        for j in range(i + 1, len(good_matches)):
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


# Conciliates the given set of match clusters <clustered_matches>, based on their image-space transformation
# agreement. The image-space information is obtained through the given sets of keypoints <keypoints1> and
# <keypoints2>. Parameters <angle_diff_tolerance> (in radians) and <dist_diff_tolerance> (in pixels) define the
# tolerance to consider two clusters equivalent in terms of transformations.
# Returns the conciliated (merged) list of match clusters.
def _conciliate_clusters(clustered_matches, keypoints1, keypoints2,
                         angle_diff_tolerance=0.174533, dist_diff_tolerance=15, scale_estim_try=10):
    # estimates the average transformation for each cluster
    transforms = []
    for match_cluster in clustered_matches:
        # estimates the difference in scale between the two matched clusters (one cluster in each image)
        scale_ratios = []
        for i in range(0, len(match_cluster) - 1):
            for j in range(i + 1, len(match_cluster)):
                # points and distance on image 1
                (x11, y11) = keypoints1[match_cluster[i].queryIdx].pt
                (x12, y12) = keypoints1[match_cluster[j].queryIdx].pt
                dist1 = math.sqrt((x11 - x12) ** 2 + (y11 - y12) ** 2)

                # points and distance on image 2
                (x21, y21) = keypoints2[match_cluster[i].trainIdx].pt
                (x22, y22) = keypoints2[match_cluster[j].trainIdx].pt
                dist2 = math.sqrt((x21 - x22) ** 2 + (y21 - y22) ** 2)

                # if the distances are not zero, estimates the difference in scale
                if dist1 > 0.0 and dist2 > 0.0:
                    scale_ratios.append(dist1 / dist2)

            if len(scale_ratios) > scale_estim_try:
                break

        if len(scale_ratios) > 0:
            scale_ratios = numpy.mean(scale_ratios)
        else:
            scale_ratios = 0.0

        # estimates the average angle and distance between the matched pairs of keypoints across images
        if scale_ratios > 0.0:
            angle_sum = 0.0
            dist_sum = 0.0

            for match in match_cluster:
                (x1, y1) = keypoints1[match.queryIdx].pt

                pt2 = keypoints2[match.trainIdx].pt
                (x2, y2) = (pt2[0] * scale_ratios, pt2[1] * scale_ratios)

                angle = math.atan2(y2 - y1, x2 - x1)
                if angle < 0.0:
                    angle = 2.0 * math.pi + angle
                angle_sum = angle_sum + angle

                distance = math.sqrt((x2 - x1) ** 2.0 + (y2 - y1) ** 2.0)
                dist_sum = dist_sum + distance

            transforms.append((angle_sum / len(match_cluster), dist_sum / len(match_cluster)))

        else:
            transforms.append(None)

    # for each pair of clusters
    for i in range(0, len(clustered_matches) - 1):
        if transforms[i] is None:
            continue

        for j in range(i + 1, len(clustered_matches)):
            if transforms[j] is None:
                continue

            if clustered_matches[i] is not None and clustered_matches[j] is not None:
                angle_i, dist_i = transforms[i]
                total_i = len(clustered_matches[i])

                angle_j, dist_j = transforms[j]
                total_j = len(clustered_matches[j])

                # if the transformations are equivalent
                if abs(angle_i - angle_j) <= angle_diff_tolerance and abs(dist_i - dist_j) <= dist_diff_tolerance:
                    # merges the current two clusters
                    clustered_matches[i].extend(clustered_matches[j])

                    # updates the resulting cluster's transformation data
                    angle_sum = angle_i * total_i + angle_j * total_j
                    dist_sum = dist_i * total_i + dist_j * total_j
                    transforms[i] = (angle_sum / (total_i + total_j), dist_sum / (total_i + total_j))

                    # cleans the old j-th cluster
                    clustered_matches[j] = None
                    transforms[j] = None

    # generates and returns the method output
    output = []
    for match_cluster in clustered_matches:
        if match_cluster is not None:
            output.append(match_cluster)
    return output


# Matches the content of image1 (<image_shape21>, <keypoints1>, <descriptions1>) with the content of image2
# (<image_shape2>, <keypoints2>, <descriptions2>), keeping geometric consistency (e.g., no matches crossing
# each other). If no information is given for image2, image1 is analysed with itself (useful to perform
# copy-move detection). Parameter <min_match_count> defines the minimum number of matches to consider a set valid.
# Returns a cluster-wise list of lists of OpenCV DMatch objects.
def match(image_shape1, keypoints1, descriptions1, image_shape2=None, keypoints2=None, descriptions2=None,
          min_match_count=9):
    if len(keypoints1) == 0 or (keypoints2 is not None and len(keypoints2) == 0):
        return []

    # defines the two sets of keypoints to be matched
    within_same_image = False
    if keypoints2 == None:
        keypoints2 = keypoints1
        descriptions2 = descriptions1
        within_same_image = True

    # performs g2nn keypoint selection
    keypoint_indices1, keypoint_indices2 = _g2nn_keypoint_selection(keypoints1, descriptions1,
                                                                    keypoints2, descriptions2,
                                                                    within_same_image)
    if len(keypoint_indices1) == 0 or len(keypoint_indices2) == 0:
        return []

    # clusterizes the selected keypoints
    selected_keypoints1 = [keypoints1[i] for i in keypoint_indices1]
    clusters1 = _clusterize_keypoints(selected_keypoints1, image_shape1)
    for c in clusters1:
        for i in range(len(c)):
            c[i] = keypoint_indices1[i]

    if within_same_image:
        clusters2 = clusters1
    else:
        selected_keypoints2 = [keypoints2[i] for i in keypoint_indices2]
        clusters2 = _clusterize_keypoints(selected_keypoints2, image_shape2)
        for c in clusters2:
            for i in range(len(c)):
                c[i] = keypoint_indices2[i]

    # for each pair of clusters
    clustered_matches = []
    for i in range(0, len(clusters1)):
        for j in range(0, len(clusters2)):
            # if within same image, i and j cannot be the same
            if within_same_image and i >= j:
                continue

            # tries to find geometrically consistent matches
            keypoints_i = [keypoints1[k] for k in clusters1[i]]
            descriptions_i = numpy.array([descriptions1[k] for k in clusters1[i]], dtype=numpy.float32)

            keypoints_j = [keypoints2[k] for k in clusters2[j]]
            descriptions_j = numpy.array([descriptions2[k] for k in clusters2[j]], dtype=numpy.float32)

            ij_matches = _consistent_match(keypoints_i, descriptions_i, keypoints_j, descriptions_j)
            for m in ij_matches:
                m.queryIdx = clusters1[i][m.queryIdx]
                m.trainIdx = clusters2[j][m.trainIdx]

            if len(ij_matches) >= min_match_count:
                clustered_matches.append(ij_matches)

    # conciliates similar clusters, regarding the estimated image-space transformations
    clustered_matches = _conciliate_clusters(clustered_matches, keypoints1, keypoints2)

    # returns the obtained clusters
    return clustered_matches
