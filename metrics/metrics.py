import json


# TODO add description.
def _count_vertices(graph):
    return len(graph['nodes'])


# TODO add description.
def _count_edges(graph):
    return len(graph['links'])


# TODO add description.
def _count_vertex_intersection(graph1, graph2):
    count = 0

    nodes1 = []
    for node in graph1['nodes']:
        id = node['file']
        if id not in nodes1:
            nodes1.append(id)
        else:
            raise Exception('There are conflicting nodes in graph 1, with file', id + '.')

    nodes2 = []
    for node in graph2['nodes']:
        id = node['file']
        if id not in nodes2:
            nodes2.append(id)
            if id in nodes1:
                count = count + 1
        else:
            raise Exception('There are conflicting nodes in graph 2, with file', id + '.')

    return count


# TODO add description.
def _count_edge_intersection(graph1, graph2, directed=True):
    count = 0

    nodes1 = {}
    for node in graph1['nodes']:
        pos = int(node['id'][2:])
        id = node['file']

        if pos not in nodes1.keys():
            nodes1[pos] = id
        else:
            raise Exception('There are conflicting nodes in graph 1, with id', str(pos) + '.')

    edges1 = []
    for edge in graph1['links']:
        source = edge['source']
        target = edge['target']

        edges1.append((nodes1[source], nodes1[target]))
        if not directed and source != target:
            edges1.append((nodes1[target], nodes1[source]))

    nodes2 = {}
    for node in graph2['nodes']:
        pos = int(node['id'][2:])
        id = node['file']

        if pos not in nodes2.keys():
            nodes2[pos] = id
        else:
            raise Exception('There are conflicting nodes in graph 2, with id', str(pos) + '.')

    for edge in graph2['links']:
        source = edge['source']
        target = edge['target']

        if (nodes2[source], nodes2[target]) in edges1:
            count = count + 1

    return count


# TODO add description.
def compute_vertex_recall(gt_graph, ans_graph):
    ans_nodes = []
    for node in ans_graph['nodes']:
        id = node['file']
        if id not in ans_nodes:
            ans_nodes.append(id)
        else:
            raise Exception('There are conflicting nodes in the answer graph, with file', id + '.')

    recall = 0
    for node in gt_graph['nodes']:
        id = node['file']
        if id in ans_nodes:
            recall = recall + 1

    return recall / float(_count_vertices(gt_graph))


# TODO add description.
def compute_vo(graph1, graph2):
    return 2.0 * (_count_vertex_intersection(graph1, graph2)) / (_count_vertices(graph1) + _count_vertices(graph2))


# TODO add description.
def compute_eo(graph1, graph2, directed=True):
    return 2.0 * (_count_edge_intersection(graph1, graph2, directed)) / (_count_edges(graph1) + _count_edges(graph2))


# TODO add description.
def compute_veo(graph1, graph2, directed=True):
    return 2.0 * (_count_vertex_intersection(graph1, graph2) + _count_edge_intersection(graph1, graph2, directed)) / \
           (_count_vertices(graph1) + _count_vertices(graph2) + _count_edges(graph1) + _count_edges(graph2))


# TODO add description.
def main(gt_graph_fp, ans_graph_fp):
    with open(gt_graph_fp) as f:
        gt_graph = json.load(f)

    with open(ans_graph_fp) as f:
        ans_graph = json.load(f)

    values = []
    values.append(compute_vertex_recall(gt_graph, ans_graph))
    values.append(compute_vo(gt_graph, ans_graph))
    values.append(compute_eo(gt_graph, ans_graph, directed=True))
    values.append(compute_veo(gt_graph, ans_graph, directed=True))
    values.append(compute_eo(gt_graph, ans_graph, directed=False))
    values.append(compute_veo(gt_graph, ans_graph, directed=False))

    return values
