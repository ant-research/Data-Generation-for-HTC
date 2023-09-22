from collections import defaultdict

label_dict = {}
hiera = defaultdict(set)
with open('data/rcv1/rcv1.taxonomy', 'r') as f:
    label_dict['Root'] = -1
    for line in f.readlines():
        line = line.strip().split('\t')
        for i in line[1:]:
            if i not in label_dict:
                label_dict[i] = len(label_dict) - 1
            hiera[label_dict[line[0]]].add(label_dict[i])
    label_dict.pop('Root')
    hiera.pop(-1)

labels = []
for item in label_dict.keys():
    labels.append(item+'\n')

f = open('./NLU_training_dataset/rcv1/labels.txt', 'w')
f.writelines(labels)
f.close()

import pdb
pdb.set_trace()