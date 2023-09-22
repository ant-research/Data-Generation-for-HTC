from rake_nltk import Rake
import json
import pdb
import yake
from tqdm import tqdm
path = "data/wos_total.json"
with open(path,'r') as f:
    lines = f.readlines()
data = []
for i in range(len(lines)):
    line = json.loads(lines[i])
    data.append(line)




language = "en" # 文档语言
max_ngram_size = 3 # N-grams
deduplication_thresold = 0.3 # 筛选阈值,越小关键词越少
deduplication_algo = 'seqm'
windowSize = 1
numOfKeywords = 20 # 最大数量

kw_extractor = yake.KeywordExtractor(lan=language, 
                                     n=max_ngram_size, 
                                     dedupLim=deduplication_thresold, 
                                     dedupFunc=deduplication_algo, 
                                     windowsSize=windowSize, 
                                     top=numOfKeywords)
                                            

label_keywords_dict = {}

cnt = 0
for line in tqdm(data):
    cnt += 1
    text = line['doc_token']
    labels = line['doc_label']
    keywords = kw_extractor.extract_keywords(text)
    #keywords -> [(token,prob)]
    for label in labels:
        if label not in label_keywords_dict:
            label_keywords_dict[label] = dict()
        else:
            for keyword in keywords:
                if keyword[0] not in label_keywords_dict[label]:
                    label_keywords_dict[label][keyword[0]] = 1
                else:
                    label_keywords_dict[label][keyword[0]] += 1


result = []
for label in label_keywords_dict:
    keywords = label_keywords_dict[label]
    tmp = []
    for keyword in keywords:
        #keywords {keyword,cnt}
        tmp.append((keyword,keywords[keyword]))
    tmp = sorted(tmp,key=lambda keyword:keyword[1],reverse=True)
    #取最高出现频率的前三百的关键词
    tmp = tmp[:300]
    # tmp = [item[0] for item in tmp]
    res = json.dumps({label:tmp})
    result.append(res)

with open("keyword_label_512/label_keywords_dict.json","w") as f:
    for line in result:
        f.write(line+'\n')


# input_y = token
# self.r.extract_keywords_from_text(token)
# raw_keyword_list = self.r.get_ranked_phrases()
# if len(raw_keyword_list) > 0:
#     mention_list = clean_top_features(raw_keyword_list, top=5)
# else:
#     current_tokens = token.split()
#     w_idf = [(w, self.idf_value[w.lower()]) for w in current_tokens if w.lower() in self.idf_value]
#     w_idf = sorted(w_idf, key=lambda x: x[1], reverse=True)
#     mention_list = [k[0] for k in w_idf[:6]]
# random.shuffle(mention_list)