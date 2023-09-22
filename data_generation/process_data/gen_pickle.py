import pickle
data = []
with open("/root/pda/pda-master/NLU_training_dataset/WebOfScience/labels.txt",'r') as f:
    data = f.readlines()

label_dict = {}
# import pdb
# pdb.set_trace()
for i in range(len(data)):
    label = data[i].strip()
    label_dict[label] = i

with open("/root/pda/pda-master/NLU_training_dataset/WebOfScience/label_dict.pkl",'wb') as f:
    pickle.dump(label_dict,f)