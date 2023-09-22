from tqdm import tqdm
import json
cnt = 0
path = "data/wos_total.json"
with open(path,'r') as f:
    lines = f.readlines()
data = []
for i in range(len(lines)):
    line = json.loads(lines[i])
    data.append(line)

label_tree = {}
for line in tqdm(data):
    cnt += 1
    text = line['doc_token']
    labels = line['doc_label']
    son = labels[1]
    fa = labels[0]
    if son in label_tree:
        continue
    else:
        label_tree[son] = fa
with open("keyword_label_512/label_tree.json",'w') as f:
    json.dump(label_tree,f)