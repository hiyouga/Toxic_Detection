import json
import random
import numpy as np
import pandas as pd


if __name__ == '__main__':
    train_df = pd.read_csv('train.csv')
    extra_df = pd.read_csv('train_extra.csv')
    dev_ratio = 0.1
    length_dict = dict()
    attrs = [
        'male',
        'female',
        'homosexual_gay_or_lesbian',
        'christian',
        'jewish',
        'muslim',
        'black',
        'white',
        'psychiatric_or_mental_illness'
    ]
    dataset = list()
    for i in range(10000):
        row = train_df.loc[i]
        extra_row = extra_df.loc[i]
        data = {
            'text': row['comment_text'],
            'target': row['target']
        }
        length = len(row['comment_text'].strip().split())
        if length in length_dict:
            length_dict[length] += 1
        else:
            length_dict[length] = 1
        for attr in attrs:
            data[attr] = extra_row[attr] if not np.isnan(extra_row[attr]) else -1
        # data['identity_annotator_count'] = int(extra_row['identity_annotator_count'])
        dataset.append(data)
    random.shuffle(dataset)
    trainset, devset = dataset[:int(len(dataset) * (1-dev_ratio))], dataset[int(len(dataset) * (1-dev_ratio)):]
    with open('train.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(trainset, sort_keys=False, indent=4))
        print(f"Processed {len(trainset)} training examples")
    with open('dev.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(devset, sort_keys=False, indent=4))
        print(f"Processed {len(devset)} development examples")
    test_df = pd.read_csv('test.csv')
    testset = list()
    for i in range(len(test_df)):
        row = test_df.loc[i]
        testset.append({'text': row['comment_text']})
    with open('test.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(testset, sort_keys=False, indent=4))
        print(f"Processed {len(testset)} test examples")
    total_num = sum(length_dict.values())
    ratio_dict = dict()
    accumulate = 0
    for i in range(max(length_dict.keys())+1):
        if i in length_dict:
            accumulate += length_dict[i]
            ratio_dict[i] = 100.0 * accumulate / total_num
