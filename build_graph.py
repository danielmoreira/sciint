# TODO add description.

import sys
import os
import json
import random
import multiprocessing

import provenance_lib.adjacency_matrix_builder as adjacency_matrix_builder
import provenance_lib.kruskal_graph_builder as kruskal_graph_builder
import provenance_lib.ip.image_reader as image_reader
import provenance_lib.ip.image_descriptor as image_descriptor
import provenance_lib.ip.pickler as pickler

with open(sys.argv[1]) as input_file:
    input_data = json.load(input_file)

probe_file_path = input_data['probe']
json_output_file_path = input_data['output']
image_file_paths = input_data['images']
image_dates = input_data['dates']

# probe in the last position
try:
    probe_index = image_file_paths.index(probe_file_path)
except ValueError:
    probe_index = -1

if probe_index > -1:
    image_file_paths.append(image_file_paths[probe_index])
    image_dates.append(image_dates[probe_index])
    del image_file_paths[probe_index]
    del image_dates[probe_index]
else:
    image_file_paths.append(probe_file_path)
    image_dates.append('00000000')

print('[INFO] There are', len(image_file_paths), 'images to be processed.')

# reads the images to be processed
images = []
for fp in image_file_paths:
    images.append(image_reader.read(fp))
    if len(images) % 10 == 0:
        print('[INFO] Read image', str(len(images)) + '/' + str(len(image_file_paths)) + '.')
print('[INFO] Read', len(images), 'images to be processed.')

# describes the images to be processed
keypoints = []
descriptions = []

pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
pool_result = [pool.apply_async(image_descriptor.describe, args=(image,)) for image in images]
pool.close()
all_descriptions = [r.get() for r in pool_result]

for i in range(len(all_descriptions)):
    current_keypoints, current_descriptions = all_descriptions[i]
    current_keypoints = pickler.unpickle_keypoints(current_keypoints)
    keypoints.append(current_keypoints[0])
    descriptions.append(current_descriptions)

print('[INFO] Described', len(keypoints), 'images.')

# generates the image adjacency matrices
adj_mat_file_path = str(random.randint(0, 999999)).zfill(6) + str(random.randint(0, 999999)).zfill(6) + \
                    str(random.randint(0, 999999)).zfill(6) + str(random.randint(0, 999999)).zfill(6) + '.npz'
adjacency_matrix_builder.build_adj_matrix(images, keypoints, descriptions, adj_mat_file_path)

print('[INFO] Created adjacency matrix.')

# generates the provenance graph
kruskal_graph_builder.json_it(image_file_paths, image_dates, adj_mat_file_path, json_output_file_path)
os.remove(adj_mat_file_path)
print('[INFO] Created provenance graph.')
