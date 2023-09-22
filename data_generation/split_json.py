import json
import pickle

with open('./keywords_balance_att.json','r') as f:
    lines = f.readlines()
keyword_list = []
for i in range(len(lines)):
    line = json.loads(lines[i])
    keyword_list.append(line)

l = len(keyword_list)//2
keyword_list_1 = keyword_list[:l]
keyword_list_2 = keyword_list[l:]

pickle.dump(keyword_list_1,open('./keywords_balance_att_1.pkl','wb'))
pickle.dump(keyword_list_2,open('./keywords_balance_att_2.pkl','wb'))

new_keyword_list_1 = pickle.load(open('./keywords_balance_att_1.pkl','rb'))
new_keyword_list_2 = pickle.load(open('./keywords_balance_att_2.pkl','rb'))

import pdb
pdb.set_trace()