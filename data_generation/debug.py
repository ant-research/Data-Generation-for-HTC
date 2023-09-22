# DataLoader(dataset, batch_size=batch_size,num_workers=0,collate_fn=collate_fn,sampler=weighted_sampler)


# DataLoader(dataset,batch_size=batch_size,num_workers=0,collate_fn=collate_fn)

import json

with open('./keywords_balance_rake_no_shuffle.json','r') as f:
    lines = f.readlines()
keyword_list = []
for i in range(len(lines)):
    line = json.loads(lines[i])
    keyword_list.append(line)

import pdb
pdb.set_trace()