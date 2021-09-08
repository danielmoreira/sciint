# TODO add description.

import io
import json


# Saves the given provenance graph in the given output file path.
# The graph is saved in NIST JSON format.
# All other parameters give additional information to generate a correct NIST JSON file.
def save_graph(graph, adj_matrix, image_filepaths, image_confidence_scores, output_filepath):
    # node info
    node_ids = []
    node_files = []
    node_scores = []

    node_count = graph.shape[0]
    for i in range(node_count):
        node_ids.append(i)
        node_files.append(image_filepaths[i])
        node_scores.append(image_confidence_scores[i])

    # json graph
    node_list = []
    edge_list = []
    connect_count = 0

    js_graph = {}
    js_graph['directed'] = True

    for i in range(node_count):
        connected = False
        for j in range(node_count):
            if graph[i, j] == 1 or graph[j, i] == 1:
                connected = True
                break

        if connected or i == 0:
            node_ids[i] = connect_count
            connect_count = connect_count + 1

            node = {}
            node['id'] = 'id' + str(node_ids[i])
            node['file'] = node_files[i]
            node['nodeConfidenceScore'] = str(node_scores[i])
            node_list.append(node)

    for i in range(node_count):
        for j in range(node_count):
            if graph[i, j] == 1:
                edge = {}
                edge['source'] = node_ids[i]
                edge['target'] = node_ids[j]
                edge['relationshipConfidenceScore'] = str(adj_matrix[i, j])
                edge_list.append(edge)

    js_graph['nodes'] = node_list
    js_graph['links'] = edge_list

    # saves JSON file
    with io.open(output_filepath, 'w', encoding='utf8') as outfile:
        graph_str = json.dumps(js_graph, indent=4, sort_keys=False, separators=(',', ':'), ensure_ascii=False)
        outfile.write(str(graph_str))
