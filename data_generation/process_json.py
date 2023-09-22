import json

with open('./keywords_balance_yake.json','r') as f:
    lines = f.readlines()
label_keywords_dict = {}
for i in range(len(lines)):
    line = json.loads(lines[i])
    import pdb
    pdb.set_trace()