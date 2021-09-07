"""
Implements an indexer and searcher of images (a.k.a, an image retriever).
"""

import sys
import os
import multiprocessing
import numpy

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/image_lib/')  # image library path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/index_lib/')  # index library path
import image_reader
import image_descriptor
import description_indexer


# Auxiliary function to perform id position binary search (id mapping).
def _id_binary_search(id, list, begin_index, end_index):
    if end_index < begin_index:
        return begin_index - 1

    else:
        ref_index = int((begin_index + end_index) / 2)

        if id == list[ref_index]:
            # ignores mapping to empty descriptions
            while ref_index + 1 < len(list) and list[ref_index] == list[ref_index + 1]:
                ref_index = ref_index + 1

            return ref_index

        elif id < list[ref_index]:
            return _id_binary_search(id, list, begin_index, ref_index - 1)

        else:
            return _id_binary_search(id, list, ref_index + 1, end_index)


# Describes a single image, if necessary (i.e., if the description file does not exist).
# Provide <add_flip_description> as TRUE, if the x-axis flipped version of the image
# must also be described; provide FALSE otherwise.
# Helps to make the description process parallel.
# Returns the path of the file where the obtained image descriptions are stored.
def _describe_image(image_file_path, add_flip_description=False):
    # defines the description file path
    description_fp = image_file_path
    if not add_flip_description:
        description_fp = description_fp + '.nflip.npy'  # no flip
    else:
        description_fp = description_fp + '.wflip.npy'  # with flip

    # does the description file already exist?
    if os.path.exists(description_fp):
        print('[WARNING] File', description_fp, 'already exists. I will not describe image again.')
        return description_fp

    # image to be described
    image = image_reader.read(image_file_path)
    if image is None:
        image = numpy.zeros((64, 64, 3), dtype=numpy.uint8)

    # image description
    keypoints, descriptions = image_descriptor.surf_detect_rsift_describe(image)  # no flip
    if len(keypoints) > 0 and add_flip_description:
        flipped_keypoints, flipped_descriptions = image_descriptor.surf_detect_rsift_describe(numpy.fliplr(image))
        keypoints = keypoints + flipped_keypoints
        descriptions = numpy.vstack((descriptions, flipped_descriptions))

    # stores the obtained descriptions
    image_descriptor.store_descriptions(keypoints, descriptions, description_fp)
    return description_fp


# Describes a list of images, given their file paths,
# and stores the obtained descriptions in the given description folder.
# Parameters:
# <image_file_paths> - The list of image file paths to be described.
# <add_flip_description> - TRUE, if the x-axis flipped versions of the images must also be described; FALSE otherwise.
# <sim_count> - Number os images described in parallel.
# Returns a list with the paths to the generated description files.
def describe_images(image_file_paths, add_flip_description=False, sim_count=multiprocessing.cpu_count()):
    # describes the given list of images with <thread_count> threads in parallel
    description_pool = multiprocessing.Pool(sim_count)
    fps = list([description_pool.apply(_describe_image, args=(fp, add_flip_description)) for fp in image_file_paths])
    description_pool.close()

    # returns the obtained list of description file paths
    return fps


# Loads the image descriptions stored in the give list of description file paths.
# Parameters:
# <description_file_paths> - The list of description file paths to be loaded into memory.
# Returns the loaded descriptions and a mapping of descriptions to description file paths.
def load_image_descriptions(description_file_paths):
    id_map = [0]

    # loads the descriptions stored in the given file paths
    descriptions = None
    for fp in description_file_paths:
        _, current_descriptions = image_descriptor.load_descriptions(fp)

        if len(current_descriptions) == 0:
            id_map.append(id_map[-1])  # repeat index, to register emptiness
            print('[WARNING] Could not obtain descriptions from', fp + '.')
        else:
            id_map.append(id_map[-1] + len(current_descriptions))

            if descriptions is None:
                descriptions = current_descriptions
            else:
                descriptions = numpy.vstack((descriptions, current_descriptions))

    # returns descriptions and index map
    return descriptions, id_map


# Indexes the given image descriptions and stores the obtained index.
# Parameters:
# <descriptions> - The descriptions to be indexed.
# <id_map> - A mapping of descriptions to image IDs.
# <index_file_path> - The path to the file that will store the generated index.
def index_image_descriptions(descriptions, id_map, index_file_path):
    # indexes the given descriptions and saves the index in <index_file_path>
    description_indexer.flat_index(descriptions, index_file_path)

    # saves the map of descriptions to images in <index_file_path>.map.npy
    numpy.save(index_file_path + '.map.npy', id_map)


# Searches the given descriptions inside the given image index.
# Parameters:
# <descriptions> - The descriptions to be queried.
# <query_id_map> - A mapping of descriptions to queried image IDs.
# <index_file_path> - The path of the file that stores the image index.
# <dataset_id_offset> - The offset to add to the dataset image IDs when performing search.
# Returns, for each given query, a list of description-wise (k-th_closest_dataset_image_id, k, distance) triples.
# The lists are grouped in a list whose size is the number of queries.
def search_images(descriptions, query_id_map, index_file_path, dataset_id_offset=0):
    # queries the given descriptions within the given image index file
    indices, distances = description_indexer.flat_search(descriptions, index_file_path)

    # loads the index mapping of dataset descriptions to indexed images
    dataset_id_map = list(numpy.load(index_file_path + '.map.npy'))

    # organizes the method output
    output = []
    for q_id in range(len(query_id_map) - 1):
        output.append([])

        for q in range(query_id_map[q_id], query_id_map[q_id + 1]):
            q_indices = indices[q]
            q_distances = distances[q]

            for i in range(len(q_indices)):
                d_index = q_indices[i]
                d_distance = q_distances[i]
                d_id = _id_binary_search(d_index, dataset_id_map, 0, len(dataset_id_map) - 1) + dataset_id_offset

                output[-1].append((d_id, i + 1, d_distance))

    # returns the obtained output
    return output


# Merges and sorts the given query-search outputs, with respect to the given list of dataset image file paths.
# Returns a list of image ranks, each with <rank_size> images.
def merge_and_sort_image_ranks(description_query_outputs, dataset_image_file_paths, rank_size=500, eps=1e-7):
    image_ranks = []

    for q in range(len(description_query_outputs[0])):  # for each query
        votes = {}
        scores = {}
        maxVote = 0.0
        maxScore = 0.0

        for o in range(len(description_query_outputs)):  # for each index part
            for t in description_query_outputs[o][q]:
                if t[0] not in votes.keys():
                    votes[t[0]] = 0.0
                    scores[t[0]] = 0.0

                votes[t[0]] = votes[t[0]] + 1.0 / (t[1])  # vote is weighted by position
                scores[t[0]] = scores[t[0]] + 1.0 / (t[2] + eps)  # score is the inverse of description distance

                if votes[t[0]] > maxVote:
                    maxVote = votes[t[0]]

                if scores[t[0]] > maxScore:
                    maxScore = scores[t[0]]

        current_rank = []  # rank of current query
        for key in votes.keys():
            score = (votes[key] / maxVote + scores[key] / maxScore) / 2.0  # average of scores
            current_rank.append((key, dataset_image_file_paths[key], score))
        current_rank = sorted(current_rank, key=lambda x: x[2], reverse=True)

        image_ranks.append(current_rank[:rank_size])

    return image_ranks
