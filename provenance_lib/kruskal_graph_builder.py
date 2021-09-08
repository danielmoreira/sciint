# TODO add description.

import numpy
import provenance_lib.graph_json_writer as graph_json_writer


class Vertex(object):
    def __init__(self, id):
        self.id = id
        self.leader = id
        self.set = [self]

    def join(self, v):
        new_leader = self.leader
        new_set = self.set + v.set

        for _v in new_set:
            _v.leader = new_leader
            del _v.set
            _v.set = new_set


def _apply_kruskal(adj_matrix, maximum_st=True):
    vertex_count = adj_matrix.shape[0]
    answer = numpy.zeros((vertex_count, vertex_count))

    vertices = []
    connected_vertex_ids = []
    for i in range(vertex_count):
        vertices.append(Vertex(i))

        for j in range(vertex_count):
            if adj_matrix[i, j] > 0 or adj_matrix[j, i] > 0:
                connected_vertex_ids.append(i)
                break

    vertex_count = len(connected_vertex_ids)

    edges = []
    for i in connected_vertex_ids:
        for j in connected_vertex_ids:
            edges.append((i, j, adj_matrix[i, j]))
    edges.sort(key=lambda e: e[2], reverse=maximum_st)

    current_edge = 0
    while vertex_count > 1:
        i = edges[current_edge][0]
        j = edges[current_edge][1]

        if vertices[i].leader != vertices[j].leader:
            answer[i, j] = 1.0
            vertices[i].join(vertices[j])
            vertex_count = vertex_count - 1

        current_edge = current_edge + 1

    return answer


# Generates a provenance DAG from a given adjacency matrix.
# Kruskal strategy (maximum spanning tree) based on the number of interest points.
def json_it(filepaths, npz_filepath, out_filepath):
    npz_file = numpy.load(npz_filepath)
    kp_matrix = npz_file['arr_0']  # number of interest points
    mi_matrix = npz_file['arr_1']  # number of interest points

    # max_kp = numpy.max(kp_matrix) + 1e-7
    # max_mi = numpy.max(mi_matrix) + 1e-7

    adj_matrix = numpy.zeros(kp_matrix.shape, dtype=numpy.float32)
    for r in range(0, adj_matrix.shape[0] - 1):
        for c in range(r + 1, adj_matrix.shape[1]):
            # adj_matrix[r, c] = 0.25 * kp_matrix[r, c] / max_kp + 0.75 * mi_matrix[r, c] / max_mi
            # adj_matrix[c, r] = 0.25 * kp_matrix[c, r] / max_kp + 0.75 * mi_matrix[c, r] / max_mi

            if mi_matrix[r, c] >= mi_matrix[c, r]:
                adj_matrix[r, c] = kp_matrix[r, c]
            else:
                adj_matrix[c, r] = kp_matrix[c, r]

    node_count = adj_matrix.shape[0]
    tree = _apply_kruskal(adj_matrix, maximum_st=True)

    # processes self-relations
    for i in range(0, kp_matrix.shape[0]):
        if kp_matrix[i, i] > 0:
            tree[i, i] = kp_matrix[i, i]

    image_file_paths = [filepaths[-1]]
    image_confidence_scores = [1.0]
    for item in filepaths:
        image_file_paths.append(item)
        image_confidence_scores.append(1.0)

        if len(image_file_paths) == node_count:
            break

    graph_json_writer.save_graph(tree, adj_matrix, image_file_paths, image_confidence_scores, out_filepath)
