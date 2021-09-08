# TODO add description.

import sys
import os
import fnmatch
import numpy

import metrics.metrics as metrics

graph_folder = sys.argv[1]

gt_graph_fps = fnmatch.filter(os.listdir(graph_folder), 'g*.json')
gt_graph_fps.sort()

directed = True

recall_values = []
vo_values = []
eo_values = []
veo_values = []
u_eo_values = []
u_veo_values = []

for gt_fp in gt_graph_fps:
    gt_fp = graph_folder + '/' + gt_fp
    ans_fp = gt_fp.replace('g', 'o')

    if not os.path.isfile(ans_fp):
        raise Exception('File', ans_fp, 'not found.')

    recall, vo, eo, veo, u_eo, u_veo = metrics.main(gt_fp, ans_fp, )
    recall_values.append(recall)
    vo_values.append(vo)
    eo_values.append(eo)
    veo_values.append(veo)
    u_eo_values.append(u_eo)
    u_veo_values.append(u_veo)

print('                  Results:', len(recall_values), 'provenance graphs.')
print('               Average NR:', numpy.mean(recall_values), '(' + str(numpy.std(recall_values)) + ')')
print('               Average VO:', numpy.mean(vo_values), '(' + str(numpy.std(vo_values)) + ')')
print('               Average EO:', numpy.mean(eo_values), '(' + str(numpy.std(eo_values)) + ')')
print('              Average VEO:', numpy.mean(veo_values), '(' + str(numpy.std(veo_values)) + ')')
print('  Average EO (undirected):', numpy.mean(u_eo_values), '(' + str(numpy.std(u_eo_values)) + ')')
print(' Average VEO (undirected):', numpy.mean(u_veo_values), '(' + str(numpy.std(u_veo_values)) + ')')
