import json
import pickle

label2ids=dict()

with open('./data/nyt/nyt_train_qd.json', 'r') as f:
        for line in f.readlines():
            line = json.loads(line)
            label_name = ', '.join(line['label_name'])
            if label_name not in label2ids:
                label2ids[label_name] = line['labels']

pickle.dump(label2ids, open('./NLU_training_dataset/nyt/label_dict.pkl','wb'))
import pdb
pdb.set_trace()