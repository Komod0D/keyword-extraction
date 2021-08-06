from preprocessing import *
import pandas as pd
from pickle import load

df = pd.read_csv("dataset_industries.csv")
print(len(df))
print(len(df["company_category_list"].dropna()))
with open("dataset_dense.npy", "rb") as f:
    dataset_dense = load(f)

tagged = df.dropna(subset=["company_category_list"]).copy()
tagged["index"] = np.arange(tagged.shape[0])

data_tags = set([tag.strip() for _, sub in tagged["company_category_list"].iteritems()
                 for tag in sub.split(",") if tag != "0" and tag != ""])

key_to_ind = {key: i for i, key in enumerate(data_tags)}
ind_to_key = {i: key for key, i in key_to_ind.items()}

all_keys = np.zeros((df.shape[0], len(key_to_ind.keys())))


for i, tags in df["company_category_list"].iteritems():
    for tag in tags.split(","):
        j = key_to_ind[tag]
        all_keys[i, j] = 1

assert np.any(all_keys > 0)
print(np.count_nonzero(all_keys.sum(axis=1) > 0))

rep_keys = repopulate_tags(all_keys.copy(), dataset_dense, df)
assert np.all(rep_keys.sum(axis=1) > 0)

indices = all_keys.sum(axis=1) > 0
missing = df[np.logical_not(indices)].copy()
missing["index"] = np.arange(missing.shape[0])
print(len(missing))

print(len(df))
print(len(df) - len(df["company_category_list"].dropna()))

for ind, _ in missing.iterrows():
    row = rep_keys[ind]
    print(row.shape)
    keys = [ind_to_key[i] for i in range(rep_keys.shape[1]) if rep_keys[i][0] > 0]
    assert len(keys) > 0
    df.at[ind, "company_category_list"] = ",".join(keys)

df.to_csv("dataset_repopulated.csv", index=False)
