import pickle

with open('./NLU_training_dataset/WebOfScience/train_whole_noise.txt') as f:
    train_whole=f.readlines()

with open('./wos_keywords_v4_25.txt') as f:
    cleaned_keywords=f.readlines()

keywords_dict={}
for i in range(len(train_whole)):
    keywords_dict[train_whole[i]]=cleaned_keywords[i]

pickle.dump(keywords_dict,open('./wos_keywords_v4_25.pkl','wb'))

keywords_dict = pickle.load(open('./wos_keywords_v4_25.pkl','rb'))

for item in keywords_dict.items():
    import pdb
    pdb.set_trace()