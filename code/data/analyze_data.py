import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    train_df = pd.read_csv('train.csv')
    extra_df = pd.read_csv('train_extra.csv')
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
    targets = list()
    safe_attr = list()
    toxic_attr = list()
    safe_attr_polarity = {k: 0 for k in attrs}
    toxic_attr_polarity = {k: 0 for k in attrs}
    safe_num, toxic_num = 0, 0
    for i in range(50000):
        row = train_df.loc[i]
        extra_row = extra_df.loc[i]
        target = row['target']
        targets.append(round(10 * target))
        for j, attr in enumerate(attrs):
            if extra_row[attr] >= 0.5:
                if target >= 0.5:
                    toxic_attr.append(j)
                    toxic_attr_polarity[attr] += 1
                    toxic_num += 1
                else:
                    safe_attr.append(j)
                    safe_attr_polarity[attr] += 1
                    safe_num += 1

    for k, p in toxic_attr_polarity.items():
        toxic_attr_polarity[k] = p / toxic_num
    for k, p in safe_attr_polarity.items():
        safe_attr_polarity[k] = p / safe_num

    plt.figure()
    arr = plt.hist(targets, bins=10)
    for i in range(10):
        plt.text(arr[1][i], arr[0][i], str(arr[0][i]))
    plt.xticks(np.arange(10))
    plt.show()

    plt.figure()
    plt.plot(np.arange(len(attrs)), safe_attr_polarity.values(), label='safe')
    plt.plot(np.arange(len(attrs)), toxic_attr_polarity.values(), label='toxic')
    plt.legend()
    plt.xticks(np.arange(len(attrs)), attrs, rotation=75)
