# with open("data/WebOfScience/wos.taxnomy",'r') as f:
#     labels_data = f.readlines()
# label_tree = dict()
# for line in labels_data:

#     labels = line.split("\t")
#     if labels[0] == 'Root':
#         continue
#     else:
#         label_tree[labels[0]] = []
#     for label in labels[1:]:
#         label_tree[labels[0]].append(label.strip('\n'))


import pandas as pd
with open("sample/text.txt",'r') as f:
    labels_data = f.readlines()

result = {'text':[]}
for item in labels_data:
    result['text'].append(item.strip('\n'))
results=pd.DataFrame(result)

results.to_csv("sample/text.csv", mode='a', index=False)

import pdb
pdb.set_trace()