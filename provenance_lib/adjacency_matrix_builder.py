# TODO add description.

import numpy
import cv2
import multiprocessing

import provenance_lib.probe_linker as probe_linker
import provenance_lib.ip.local_similarity_analyser as local_similarity_analyser
import provenance_lib.ip.pickler as pickler


# TODO add description.
def _conciliate_outputs(values):
    all_matches = pickler.unpickle_all_matches(values[0][4])
    flip_info = values[0][5]

    for i in range(1, len(values)):
        i_all_matches = pickler.unpickle_all_matches(values[i][4])
        i_flip_info = values[i][5]

        for j in range(0, len(flip_info)):
            if i_flip_info[j] != -2:
                all_matches[j] = i_all_matches[j]
                flip_info[j] = i_flip_info[j]

    return all_matches, flip_info


# TODO add description.
def _compute_ij_adj_matrix(i, j, keypoints_with_probe_last, descriptions_with_probe_last, rank_with_probe_last,
                           all_matches, flip_info):
    keypoints_with_probe_last = pickler.unpickle_keypoints(keypoints_with_probe_last)
    all_matches = pickler.unpickle_all_matches(all_matches)

    # obtains the computed matches between images i and j
    image_count = len(rank_with_probe_last)
    p1 = image_count * (i + 1) + (j + 1)
    p2 = image_count * (j + 1) + (i + 1)
    matches = all_matches[p1]
    flip = flip_info[p1]  # should be the same as flip_info[p2]

    # if matches were not calculated yet...
    if flip == -2:
        matches, flip = probe_linker.compute_ij_relation(i, j, pickler.pickle_keypoints(keypoints_with_probe_last),
                                                         descriptions_with_probe_last, rank_with_probe_last)
        matches = pickler.unpickle_matches(matches)
        all_matches[p1] = all_matches[p2] = matches[:]
        flip_info[p1] = flip_info[p2] = flip

    # if there are matches
    if len(matches) > 0:
        image_i = rank_with_probe_last[i]
        image_j = rank_with_probe_last[j]
        if flip == 1:
            image_j = cv2.flip(image_j, 1)

        kp_count, mi_ij, mi_ji = local_similarity_analyser.compare(image_i, keypoints_with_probe_last[i][0],
                                                                   image_j, keypoints_with_probe_last[j][flip],
                                                                   matches)

        return True, kp_count, mi_ij, mi_ji, pickler.pickle_all_matches(all_matches), flip_info[:]

    return False, 0.0, 0.0, 0.0, pickler.pickle_all_matches(all_matches), flip_info[:]


# Builds N provenance adjacency matrices based on:
#    i. keypoint count;
#    ii. mutual information.
# Saves the matrices in the given output file path.
def build_adj_matrix(rank_with_probe_last, keypoints_with_probe_last, descriptions_with_probe_last, output_filepath):
    image_count = len(rank_with_probe_last)

    # obtains only the images that are related to the probe
    related_images, all_matches, flip_info = probe_linker.get_related_images_to_probe(rank_with_probe_last,
                                                                                      keypoints_with_probe_last,
                                                                                      descriptions_with_probe_last)
    print('[DEBUG]', related_images)

    # adjacency matrices
    kp_count_mat = numpy.zeros((image_count, image_count))
    mut_info_mat = numpy.zeros((image_count, image_count))
    # other matrices can be added here...

    # for each pair of related images...
    for l in range(0, len(related_images) - 1):
        # organizes sources and target images
        sources = []
        targets = []

        for m in range(0, len(related_images[l])):
            sources.append(related_images[l][m])

        for m in range(0, len(related_images[l + 1])):
            sources.append(related_images[l + 1][m])
            targets.append(related_images[l + 1][m])

        # for each pair source-target
        for i in sources:
            current_js = []
            for j in targets:
                if i != j and (i not in targets or i < j):
                    current_js.append(j)

            if len(current_js) > 0:
                print('[DEBUG] Processing', i, 'against', current_js)
                pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
                pool_result = [pool.apply_async(_compute_ij_adj_matrix,
                                                args=(i, j, pickler.pickle_keypoints(keypoints_with_probe_last),
                                                      descriptions_with_probe_last, rank_with_probe_last,
                                                      pickler.pickle_all_matches(all_matches), flip_info))
                               for j in current_js]
                pool.close()
                values = [r.get() for r in pool_result]

                if len(values) > 0:
                    all_matches, flip_info = _conciliate_outputs(values)

                for v in range(len(values)):
                    if values[v][0]:
                        a = i + 1
                        b = current_js[v] + 1

                        kp_count_mat[a, b] = kp_count_mat[b, a] = values[v][1]
                        mut_info_mat[a, b] = numpy.max(values[v][2])
                        mut_info_mat[b, a] = numpy.max(values[v][3])

    # saves everything
    numpy.savez(output_filepath, kp_count_mat, mut_info_mat)
