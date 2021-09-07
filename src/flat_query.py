import sys
import os

import flat_image_retriever

##############
# Input parameters (there are 3)
##############
# 1. Query image file path.
query_file_path = sys.argv[1]

# 2. Dataset image list file path.
dataset_list = sys.argv[2]

# 3. Output rank file path.
output_file_path = sys.argv[3]

##############
# Configuration parameters
##############
RANK_SIZE = 500

##############
# Main process execution
##############
print('[INFO] Usage: python3 flat_query.py <query_file_path> <dataset_folder> <output_file_path>')
print('[INFO] Called: python3 flat_query.py', query_file_path, dataset_list, output_file_path)

# A. Obtains the paths to the images to be processed.
image_file_paths = []
with open(dataset_list) as f:
    for line in f:
        image_file_paths.append(line.strip())
print('[INFO] Query', query_file_path, 'will be compared to', len(image_file_paths), 'images.')

# if the image index does not exist
index_fp = 'flat_index.fai'
if not os.path.exists(index_fp):
    print('[INFO] Image index does not exist. I will build it now.')

    # image index needs to be built
    # B. Describes the images to be processed.
    description_files = flat_image_retriever.describe_images(image_file_paths, add_flip_description=True)
    print('[INFO]    Described all images.')

    # C. Loads the descriptions of the images to be indexed.
    descriptions, id_map = flat_image_retriever.load_image_descriptions(description_files)
    print('[INFO]    Loaded all image descriptions.')

    # D. Indexes the images to be processed.
    # index_fp = os.path.join(description_folder, 'flat_index.fai')
    flat_image_retriever.index_image_descriptions(descriptions, id_map, index_fp)
    print('[INFO]    Indexed dataset images.')

# else, no need to build it again
else:
    print('[INFO] Image index already exists. I will not build it again.')

# E. Describes the query.
query_description_file = flat_image_retriever.describe_images([query_file_path], add_flip_description=False)
print('[INFO] Described the query image.')

# F. Loads the query description.
query_descriptions, query_id_map = flat_image_retriever.load_image_descriptions(query_description_file)
print('[INFO] Loaded query image descriptions.')

# G. Retrieves the related query images from the index file.
query_output = flat_image_retriever.search_images(query_descriptions, query_id_map, index_fp, dataset_id_offset=0)
print('[INFO] Performed image query.')

# H. Merges the outputs and stores the resulting image rank.
image_rank = flat_image_retriever.merge_and_sort_image_ranks([query_output], image_file_paths, rank_size=RANK_SIZE)[0]

with open(output_file_path, 'w') as f:
    for item in image_rank:
        # f.write(item[1].split('/')[-1] + ',' + "{:.10f}".format(item[2]) + '\n')
        f.write(item[1] + ',' + "{:.10f}".format(item[2]) + '\n')

print('[INFO] Query', query_file_path + ':', 'generated and stored image rank.')

print('*** Acabou. *** ')
