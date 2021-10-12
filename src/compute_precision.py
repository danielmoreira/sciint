import json
import numpy

for n in [1, 5, 10]:
    precisions = []

    gt_dict = {}
    with open('../ranking_data/gt-ranks.json') as f:
        data = json.load(f)
        for item in data:
            gt_dict[item] = data[item]['rank'][:]

    sl_dict = {}
    with open('../ranking_data/rank_list.txt') as f:
        for file_path in f:
            id = file_path.strip().replace('rank_images/', '').replace('.txt', '')
            sl_dict[id] = []
            # print('Rank ID:', id)

            with open(file_path.strip()) as rf:
                count = 0
                for line in rf:
                    parts = line.strip().split(',')
                    if len(parts) == 2:
                        retrieved_id = parts[0].replace('rank_images/', '')
                        if retrieved_id != id:
                            sl_dict[id].append(retrieved_id)
                            count = count + 1
                            if count >= n:
                                break

    for key in sl_dict.keys():
        # print(key)

        relevant_count = 0.0
        for retrieved_id in sl_dict[key]:
            if retrieved_id in gt_dict[key]:
                relevant_count = relevant_count + 1.0

        precisions.append(relevant_count / n)

    print('***********************************')
    print('P@' + str(n))
    # print('P list:', precisions)
    print('P  max:', numpy.max(precisions))
    print('P  min:', numpy.min(precisions))
    print('P mean:', numpy.mean(precisions))
    print('P stdv:', numpy.std(precisions))
    print('***********************************')
