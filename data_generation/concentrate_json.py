    with open('./keywords_balance_rake_shuffle_class_keywords.json','r') as f:
        lines = f.readlines()
    self.keyword_list = []
    for i in range(len(lines)):
        line = json.loads(lines[i])