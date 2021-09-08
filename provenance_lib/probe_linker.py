# TODO add description.

import multiprocessing

import provenance_lib.ip.pickler as pickler
import provenance_lib.ip.keypoint_matcher as keypoint_matcher


# Computes the matches between images i and j within the list of images, given their keypoints and descriptions.
def compute_ij_relation(i, j, keypoints, descriptions, imgs):
    keypoints = pickler.unpickle_keypoints(keypoints)

    if imgs[i] is None or imgs[j] is None:
        return [], -1

    if len(keypoints[i]) == 0 or len(keypoints[j]) == 0:
        return [], -1

    if i == j:  # self-relation, copy-move detection
        normal_matches = keypoint_matcher.match(imgs[i].shape, keypoints[i][0], descriptions[i][0])
    else:
        normal_matches = keypoint_matcher.match(imgs[i].shape, keypoints[i][0], descriptions[i][0],
                                                imgs[j].shape, keypoints[j][0], descriptions[j][0])

    flip_matches = keypoint_matcher.match(imgs[i].shape, keypoints[i][0], descriptions[i][0],
                                          imgs[j].shape, keypoints[j][1], descriptions[j][1])

    matches = normal_matches
    flip = 0
    if len(matches) < len(flip_matches):
        matches = flip_matches
        flip = 1

    print('[DEBUG] Linking', i, j)
    return pickler.pickle_matches(matches), flip


# Traces back the rank images that might be connected to a probe,
# based on the geometrically consistent matches between them.
def get_related_images_to_probe(rank_images_with_probe_last, keypoints_with_probe_last, descriptions_with_probe_last):
    image_count = len(rank_images_with_probe_last)

    # answers of the method
    probe_related_image_indices = []
    matches = []
    flip_info = []
    for i in range(image_count * image_count):
        matches.append([])
        flip_info.append(-2)

    # current sources
    source_image_file_indices = [-1]  # starts with the probe as source

    # current targets
    target_image_file_indices = list(range(image_count - 1))  # starts with all other images as target

    # while there are available sources...
    while len(source_image_file_indices) > 0:
        probe_related_image_indices.append(source_image_file_indices[:])

        # simplified execution (direct graph only)
        # if len(probe_related_image_indices) > 1:
        #    break
        # end of simplified execution

        current_source_image_file_indices = source_image_file_indices[:]
        del source_image_file_indices[:]

        for i in current_source_image_file_indices:
            current_target_image_file_indices = target_image_file_indices[:]

            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            pool_result = [pool.apply_async(compute_ij_relation,
                                            args=(i, j, pickler.pickle_keypoints(keypoints_with_probe_last),
                                                  descriptions_with_probe_last, rank_images_with_probe_last))
                           for j in current_target_image_file_indices]
            pool.close()
            relations = [r.get() for r in pool_result]

            for p in range(len(current_target_image_file_indices)):
                current_matches = pickler.unpickle_matches(relations[p][0])

                j = current_target_image_file_indices[p]
                if len(current_matches) > 0:
                    source_image_file_indices.append(j)
                    target_image_file_indices.remove(j)

                p1 = image_count * (i + 1) + (j + 1)
                p2 = image_count * (j + 1) + (i + 1)
                matches[p1] = matches[p2] = current_matches[:]
                flip_info[p1] = flip_info[p2] = relations[p][1]

            del current_target_image_file_indices[:]

        del current_source_image_file_indices[:]

    return probe_related_image_indices, matches, flip_info
