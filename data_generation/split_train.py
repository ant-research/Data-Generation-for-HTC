# import random



# # random.shuffle(origin)

# sample = open('./sample.txt', 'r').readlines()



# text = []
# label = []

# for item in sample:
#     text.append(item.split('\t')[0]+'\n')
#     label.append(item.split('\t')[1])


# f = open('./text.txt', 'w')
# f.writelines(text)
# f.close()


# f = open('./label.txt', 'w')
# f.writelines(label)
# f.close()

origin = open('NLU_training_dataset/WebOfScience/train_whole_noise.txt', 'r').readlines()

f = open('./train_whole_noise_1.txt', 'w')
f.writelines(origin[:len(origin)//4])
f.close()

f = open('./train_whole_noise_2.txt', 'w')
f.writelines(origin[len(origin)//4:len(origin)//2])
f.close()

f = open('./train_whole_noise_3.txt', 'w')
f.writelines(origin[len(origin)//2:(len(origin)//4)*3])
f.close()

f = open('./train_whole_noise_4.txt', 'w')
f.writelines(origin[(len(origin)//4)*3:])
f.close()