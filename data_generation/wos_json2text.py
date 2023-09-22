import json


train=[]

with open('./data/wos/wos_train_noise.json', 'r') as f:
        for line in f.readlines():
            try:
                line = json.loads(line)
            except:
                import pdb
                pdb.set_trace()
            train.append('\t'.join([line['doc_token'],', '.join(line['doc_label'])])+'\n')


f = open('./NLU_training_dataset/WebOfScience/train_whole_noise.txt', 'w')
f.writelines(train)
f.close()

valid=[]

with open('./data/wos/wos_val_noise.json', 'r') as f:
        for line in f.readlines():
            line = json.loads(line)
            valid.append('\t'.join([line['doc_token'],', '.join(line['doc_label'])])+'\n')


f = open('./NLU_training_dataset/WebOfScience/valid_whole_noise.txt', 'w')
f.writelines(valid)
f.close()

test=[]

with open('./data/wos/wos_test_noise.json', 'r') as f:
        for line in f.readlines():
            line = json.loads(line)
            test.append('\t'.join([line['doc_token'],', '.join(line['doc_label'])])+'\n')


f = open('./NLU_training_dataset/WebOfScience/test_whole_noise.txt', 'w')
f.writelines(test)
f.close()



import pdb
pdb.set_trace()