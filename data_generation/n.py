
labels = []
with open('NLU_training_dataset/nyt/train_whole.txt') as out:

    for l in out:
        l = l.strip()
        items = l.split('\t')[1].split(', ')
        for item in items:
            if item not in labels:
                labels.append(item)
with open('NLU_training_dataset/nyt/valid_whole.txt') as out:

    for l in out:
        l = l.strip()
        items = l.split('\t')[1].split(', ')
        for item in items:
            if item not in labels:
                labels.append(item)
with open('NLU_training_dataset/nyt/test_whole.txt') as out:

    for l in out:
        l = l.strip()
        items = l.split('\t')[1].split(', ')
        for item in items:
            if item not in labels:
                labels.append(item)                

new_labels = []
for item in labels:
    new_labels.append(item+'\n')

f = open('./NLU_training_dataset/nyt/labels.txt', 'w')
f.writelines(new_labels)
f.close()


import pdb
pdb.set_trace()

    