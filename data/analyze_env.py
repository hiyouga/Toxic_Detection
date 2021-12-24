import pandas as pd

group2subgroups = {
    "disability": ["[no_disability]", "physical_disability", "psychiatric_or_mental_illness", "intellectual_or_learning_disability",
                   "other_disability"],
    "gender": ["[no_gender]", "male", "female", "transgender", "other_gender"],
    "sexual_orientation": ["[no_sexual]", "homosexual_gay_or_lesbian", "heterosexual", "bisexual", "other_sexual_orientation"],
    "race_or_ethnicity": ["[no_race]", "black", "white", "asian", "latino", "jewish", "other_race_or_ethnicity"],
    "religion": ["[no_religion]", "christian", "buddhist", "atheist", "muslim", "hindu", "other_religion"],
    # "misc": ["disagree", "sad", "funny", "wow", "likes"],
}


def row2envs_id(ex_row: pd.Series):
    envs_id = []
    for g, subs in group2subgroups.items():
        found = 0
        for ind, sub in enumerate(subs[1:]):
            if pd.notna(ex_row[sub]) and ex_row[sub] > 0.5:
                found += 1
                envs_id.append(1 + ind)
                break
        if not found:
            envs_id.append(0)
    return envs_id


if __name__ == '__main__':
    def valid_pos(p):
        if p > 0.5:
            return 1
        return 0


    import numpy as np
    from matplotlib import pyplot as plt
    from collections import defaultdict

    subgroup2group = {}
    for g, subs in group2subgroups.items():
        for sub in subs:
            subgroup2group[sub] = g

    t_path = r"train.csv"
    e_path = r"train_extra.csv"
    train_df = pd.read_csv(t_path)
    extra_df = pd.read_csv(e_path)

    multi_label_id = set()
    subgroup2cnt = defaultdict(lambda: defaultdict(int))
    for i in range(100000):
        row = train_df.loc[i]
        extra_row = extra_df.loc[i]
        r_id = row['id']
        e_id = extra_row['id']
        assert r_id == e_id
        target = valid_pos(row['target'])
        for g, subs in group2subgroups.items():
            found = 0
            for sub in subs[1:]:
                if pd.notna(extra_row[sub]) and valid_pos(extra_row[sub]):
                    found += 1
                    subgroup2cnt[sub][target] += 1
            if not found:
                subgroup2cnt[subs[0]][target] += 1
            if found > 1:
                multi_label_id.add(r_id)

    for g, subs in group2subgroups.items():
        sub2total = {}
        sub2pos_ratio = {}
        for sub in subs:
            total = sum(subgroup2cnt[sub].values())
            sub2total[sub] = total
            sub2pos_ratio[sub] = subgroup2cnt[sub][1] / (total + 1e-9)
        total = sum(sub2total.values())
        sub2ratio = {k: v / total for k, v in sub2total.items()}
        sub2pos_total_ratio = {}
        for sub in subs:
            sub2pos_total_ratio[sub] = sub2ratio[sub] * sub2pos_ratio[sub]

        # keys = [k for k in sub2ratio if '[' not in k]
        keys = [k for k in sub2ratio]
        index = np.asarray(list(range(len(keys))))
        ratio = [sub2ratio[keys[i]] for i in index]
        local_pos_ration = [sub2pos_ratio[keys[i]] for i in index]
        pos_ratio = [sub2pos_total_ratio[keys[i]] for i in index]
        fig = plt.figure(figsize=(4, 8))
        print(keys)
        print(sub2ratio)
        print(sub2pos_ratio)
        print(sub2pos_total_ratio)

        # filter the [no_XXX] subgroup away from showing ratio
        # plt.bar(x=2 * index[1:], height=ratio[1:], width=0.2, color='yellow', label='ratio')
        # plt.bar(x=2 * index, height=pos_ratio, width=0.2, color='green', label='pos_ratio')
        plt.title(g)
        plt.bar(x=2 * index + 1, height=local_pos_ration, width=0.2, color='blue', label='local_pos_ratio')
        plt.xticks(2 * index + 0.5, keys, rotation=15)
        plt.show()
        _ = input()
