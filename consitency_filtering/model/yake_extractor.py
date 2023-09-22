import yake
language = "en" # 文档语言
max_ngram_size = 3 # N-grams
deduplication_thresold = 0.3 # 筛选阈值,越小关键词越少
deduplication_algo = 'seqm'
windowSize = 1
numOfKeywords = 50 # 最大数量


def get_extractor():
    extractor = kw_extractor = yake.KeywordExtractor(lan=language, 
                                     n=max_ngram_size, 
                                     dedupLim=deduplication_thresold, 
                                     dedupFunc=deduplication_algo, 
                                     windowsSize=windowSize, 
                                     top=numOfKeywords)

    return extractor