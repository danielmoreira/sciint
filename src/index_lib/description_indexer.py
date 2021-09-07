"""
Implements an indexer and searcher of feature vectors (a.k.a., descriptions).
"""

import time
import numpy
import faiss


# Builds an index with the given <descriptions> and stores it in <index_file_path>.
# No compression, no loss, no optimizations.
def flat_index(descriptions, index_file_path):
    # number of descriptions
    description_count = len(descriptions)
    if description_count == 0:
        raise Exception('[ERROR] There are no descriptions to be indexed.')

    # size of descriptions
    description_size = descriptions.shape[1]

    # flat index
    index = faiss.IndexFlatL2(description_size)
    index.add(descriptions)
    faiss.write_index(index, index_file_path)


# Builds an OPQ index with the given <descriptions> and stores it in <index_file_path>.
# Other parameters:
# <coarse_index_size> - Number of centroids in the coarse index.
# <opq_pca_size> - Size of the descriptions after PCA application.
# <pq_index_div_factor> - PQ division factor. It must be a sub-multiple of <opq_pca_size>.
# <opq_train_size_factor> - Portion of descriptions that will be used to train the OPQ matrix.
# <index_bit_size> - Number of bits used for each index.
# <random_seed> - Seed of the random number generator, to provide reproducibility.
# <gpu_id> - ID of the GPU to be used, -1 if CPU execution.
def opq_index(descriptions, index_file_path, coarse_index_size, opq_pca_size, pq_index_div_factor,
              opq_train_size_factor=0.1, index_bit_size=8, random_seed=time.time(), gpu_id=-1):
    # number of descriptions
    description_count = len(descriptions)
    if description_count == 0:
        raise Exception('[ERROR] There are no descriptions to be indexed.')

    # size of descriptions
    description_size = descriptions.shape[1]
    if opq_pca_size > description_size:
        raise Exception('[ERROR] <opq_pca_size> larger than the description size. pca:', str(opq_pca_size) + ', d:',
                        str(description_size) + '.')
    if opq_pca_size % pq_index_div_factor != 0:
        raise Exception('[ERROR] <opq_pca_size> is not a multiple of <pq_index_div_factor>. pca:',
                        str(opq_pca_size) + ', pq:', str(pq_index_div_factor) + '.')

    # sets the numpy random seed
    numpy.random.seed(random_seed)

    # samples (<opq_train_size_factor> * 100)% of descriptors for learning OPQ data transformation
    train_size = int(round(description_count * opq_train_size_factor))
    if train_size <= 0 or train_size > description_count:
        train_descriptions = descriptions
    else:
        random_indices = numpy.random.randint(description_count, size=train_size)
        train_descriptions = descriptions[random_indices, :]

    # creates the OPQ index...
    # OPQ transformation matrix
    opq_matrix = faiss.OPQMatrix(description_size, pq_index_div_factor, opq_pca_size)

    # coarse (k-means), PQ, and OPQ indices
    coarse_idx = faiss.IndexFlatL2(opq_pca_size)
    pq_idx = faiss.IndexIVFPQ(coarse_idx, opq_pca_size, coarse_index_size, pq_index_div_factor, index_bit_size)
    opq_idx = faiss.IndexPreTransform(opq_matrix, pq_idx)

    # if this method shall run on CPU
    if gpu_id < 0 or gpu_id >= faiss.get_num_gpus():
        # OPQ index training and data addition
        opq_idx.train(train_descriptions)
        opq_idx.add(descriptions)

        # saves the index in the proper output file
        faiss.write_index(opq_idx, index_file_path)

    # else, makes it run on GPU
    else:
        gpu = faiss.StandardGpuResources()
        gpu_idx = faiss.index_cpu_to_gpu(gpu, gpu_id, opq_idx)

        # index training and data addition
        gpu_idx.train(train_descriptions)
        gpu_idx.add(descriptions)

        # saves the index in the proper output file
        faiss.write_index(faiss.index_gpu_to_cpu(gpu_idx), index_file_path)


# Searches a given list of row-wise <descriptions> within the index stored in <index_file_path>.
# Other parameters:
# <knn> - Number of nearest indexed descriptions to be retrieved.
# Returns a pair of lists: (1) the <knn> indices of the nearest descriptions, and (2) the <knn> respective distances.
def flat_search(descriptions, index_file_path, knn=8):
    index = faiss.read_index(index_file_path)
    distances, indices = index.search(descriptions, knn)
    return indices, distances


# Searches a given list of row-wise <descriptions> within the OPQ index stored in <index_file_path>.
# Other parameters:
# <knn> - Number of nearest indexed descriptions to be retrieved.
# <gpu_id> - ID of the GPU to be used, -1 if CPU execution.
# Returns a pair of lists: (1) the <knn> indices of the nearest descriptions, and (2) the <knn> respective distances.
def opq_search(descriptions, index_file_path, knn=8, gpu_id=-1):
    index = faiss.read_index(index_file_path)

    if gpu_id < 0 or gpu_id >= faiss.get_num_gpus():
        distances, indices = index.search(descriptions, knn)
        return indices, distances

    else:
        gpu = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(gpu, gpu_id, index)
        distances, indices = gpu_index.search(descriptions, knn)
        return indices, distances
