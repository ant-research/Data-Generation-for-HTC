import copy
import random

data_list = {}


out = open('./NLU_training_dataset/WebOfScience/train_whole.txt','r').readlines()



for l in out:

    items = l.split('\t')
    if not len(items) == 2: continue
    if items[1] not in data_list:
        data_list[items[1]]=[l]
    else:
        data_list[items[1]].append(l)



data_num={}

for key in data_list.keys():
    data_num[key] = len(data_list[key])

max_num = max(data_num.values())

final_data_list = []

for item in data_list.items():
    data = copy.deepcopy(item[1])
    random.shuffle(data)
    final_data_list.extend(data*(max_num//data_num[item[0]])+data[:max_num-(max_num//data_num[item[0]])*data_num[item[0]]])



f = open('./NLU_training_dataset/WebOfScience/train_whole_balance.txt', 'w')
f.writelines(final_data_list)
f.close()


data = open('./NLU_training_dataset/WebOfScience/train_whole_balance.txt','r').readlines()

import pdb
pdb.set_trace()