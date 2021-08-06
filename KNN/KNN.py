import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity

from Load import *

print("Loading and preprocessing data...")
dataset = pd.read_csv("dataset.csv")
corpus = dataset["company_short_description"].dropna().to_list()
corpus += dataset["company_description"].dropna().to_list()

cv = CountVectorizer()
word_counts = cv.fit_transform(corpus)
num_features = word_counts.shape[1]
tfidf = TfidfTransformer()

ours = cv.transform(dataset["text"].tolist())
sparse = tfidf.fit_transform(ours)

tagged = dataset.dropna(subset=["company_category_list"])

data_tags = set([tag.strip() for _, sub in tagged["company_category_list"].iteritems()
                 for tag in sub.split(",") if tag != "0"])

print(data_tags)

print("Dimensionality reduction...")
svd = TruncatedSVD(n_iter=10, n_components=500)
svd.fit(sparse)
print(np.sum(svd.explained_variance_ratio_))
input("Is that acceptable?")

X_test = svd.transform(tfidf.transform(cv.transform(tagged["text"].tolist())))
X_out = svd.transform(tfidf.transform(cv.transform(dataset["text"].tolist())))
tagged["indices"] = np.arange(tagged.shape[0])

labels = {key: i for i, key in enumerate(data_tags)}
inverse = {i: key for key, i in labels.items()}
Y_test = np.zeros((tagged.shape[0], len(labels.keys())))
for i, ts in tagged["company_category_list"].reset_index(drop=True).iteritems():
    for t in ts.split(","):
        t = t.strip()
        if t in data_tags:
            Y_test[i, labels[t]] = 1
        else:
            print("FATAL ERROR")
            print(t)
            input("FATAL ERROR")


similarities = cosine_similarity(X_out)
keys = np.zeros_like(similarities)

reliability = np.zeros_like(similarities)

tindices = set(tagged.index)
for i in range(tagged.shape[0]):
    for j in range(tagged.shape[0]):
        if i in tindices and j in tindices:
            x = tagged.loc[i, "indices"]
            y = tagged.loc[j, "indices"]
            temp = np.dot(Y_test[i], Y_test[j]) / len(data_tags)
            keys[x, y] = np.power(temp, 1.0 / 5)


out_df = pd.DataFrame(similarities, index=X_out["company_uuid"], columns=X_out["company_uuid"])
out_df.to_csv("similarities.csv")

knn = KNeighborsClassifier(algorithm="auto", n_neighbors=5, weights="distance")
print("fitting")
knn.fit(X_test, Y_test)
Y_pred = knn.predict_proba(X_test)

print("Predicting")
Y_out = knn.predict_proba(X_out)
for i in range(len(Y_out)):
    Y_out[i] = Y_out[i][:, 1].reshape((Y_out[i].shape[0], 1))
Y_out = np.hstack(Y_out)

ind_out = np.zeros_like(Y_out)
for i in range(Y_out.shape[0]):
    ind_out[i] = np.flip(np.argsort(Y_out[i]))

all_keys = []
for i in range(ind_out.shape[0]):
    all_keys.append([inverse[ind_out[i, j]] for j in range(3)])

for i, key in enumerate(all_keys):
    print(dataset.iloc[i]["text"])
    print(key)
    if input("next ") == "ok":
        break

tags = pd.DataFrame(all_keys, columns=["key1", "key2", "key3"], index=dataset["ivc_number"].to_list())

tags.to_csv("companies_repopulated_knn.csv", index=True)
