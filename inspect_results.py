import pandas as pd
import numpy as np

PATH = "restrict_tag"
df = pd.read_csv(f"{PATH}/crunchbase_sample.csv")
sims = np.load(f"{PATH}/cosine_sample.npy", allow_pickle=True)
keys = np.load(f"{PATH}/keys_sample.npy", allow_pickle=True)
rels = np.load(f"{PATH}/reliabilities.npy", allow_pickle=True)

both = (sims + keys) / 2

ninetieth = np.quantile(both, 0.95)
print(f"Mean similarity: {both.mean()}")
print(f"Median similarity: {np.median(both)}")
print(f"95th Percentile: {ninetieth}")

pos_ex = {"text1": [], "keywords1": [], "text2": [], "keywords2": [], "cosine": [], "keyword_similarity": [],
          "reliability": [], "total_similarity": []}

l, r = np.nonzero(both > ninetieth)
assert np.all(both[l, r] > ninetieth)

print("Generating Positive examples:")
inds = np.random.choice(l.shape[0], size=1000, replace=False)
for t in inds:
    i, j = l[t], r[t]
    sim = sims[i, j]
    rel = rels[i, j]
    key = keys[i, j]
    bot = both[i, j]
    assert bot > ninetieth
    pos_ex["text1"].append(df["text"][i])
    pos_ex["text2"].append(df["text"][j])

    pos_ex["keywords1"].append(df["company_category_list"][i])
    pos_ex["keywords2"].append(df["company_category_list"][j])

    pos_ex["cosine"].append(sim.item(0))
    pos_ex["keyword_similarity"].append(key.item(0))
    pos_ex["reliability"].append(rel.item(0))
    pos_ex["total_similarity"].append(bot.item(0))

print("Saving Positive Examples")
pos_df = pd.DataFrame.from_dict(pos_ex)
pos_df.to_csv("positive_examples.csv", index=False)


print("Generating negative examples:")
neg_ex = {"text1": [], "keywords1": [], "text2": [], "keywords2": [], "cosine": [], "keyword_similarity": [],
          "reliability": [], "total_similarity": []}

l, r = np.nonzero(both < ninetieth)
inds = np.random.choice(l.shape[0], size=1000, replace=False)
for t in inds:
    i, j = l[t], r[t]
    sim = sims[i, j]
    rel = rels[i, j]
    key = keys[i, j]
    bot = both[i, j]
    neg_ex["text1"].append(df["text"][i])
    neg_ex["text2"].append(df["text"][j])

    neg_ex["keywords1"].append(df["company_category_list"][i])
    neg_ex["keywords2"].append(df["company_category_list"][j])

    neg_ex["cosine"].append(sim.item(0))
    neg_ex["keyword_similarity"].append(key.item(0))
    neg_ex["reliability"].append(rel.item(0))
    neg_ex["total_similarity"].append(bot.item(0))

print("Saving Negative Examples")
neg_df = pd.DataFrame.from_dict(neg_ex)
neg_df.to_csv("negative_examples.csv", index=False)