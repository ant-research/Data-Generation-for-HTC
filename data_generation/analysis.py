import json

with open('label_keywords_dict.json','rb') as f:
    lines = f.readlines()

label_keywords_dict = {}
for i in range(len(lines)):
    line = json.loads(lines[i])
    label = list(line.keys())[0]
    label_keywords_dict[label] = line[label]

label2key={}

for key in label_keywords_dict.keys():

    label2key[key] = len(label_keywords_dict[key])

import pdb
pdb.set_trace()