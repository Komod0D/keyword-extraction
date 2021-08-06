import time
from pickle import dump

import h5py
import numpy as np
from nltk.corpus import words, stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
import re

stop_eng = stopwords.words("english")


def preprocess(s):
    return re.sub(r"[^a-zA-z0-9\s]", " ", s).lower()


def generate_ngrams(tokens, n=3):
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]


def tokenize(s, vocab):
    return [token for token in re.findall(r"\b\w+\b", s) if token in vocab and token not in stop_eng]


def vectorize(dataset):
    vocab = set(w.lower() for w in words.words())
    cv = CountVectorizer(min_df=5, max_df=0.25, vocabulary=vocab, stop_words=stopwords.words("english"))
    tfidf = TfidfTransformer()

    dataset_counts = cv.fit_transform(dataset["text"].tolist())
    dataset_sparse = tfidf.fit_transform(dataset_counts)

    print("dimensionality reduction")
    svd = TruncatedSVD(n_iter=10, n_components=200)
    dataset_dense = svd.fit_transform(dataset_sparse)
    with open("dataset_dense.npy", "wb") as f:
        dump(dataset_dense, f)

    return dataset_dense


def encode_tags(dataset):
    tagged = dataset.dropna(subset=["company_category_list"]).copy()
    tagged["index"] = np.arange(tagged.shape[0])

    data_tags = set([tag.strip() for _, sub in tagged["company_category_list"].iteritems()
                     for tag in sub.split(",") if tag != "0" and tag != ""])

    labels = {key: i for i, key in enumerate(data_tags)}

    all_keys = np.zeros((dataset.shape[0], len(labels.keys())))

    for i, tags in dataset["company_category_list"].dropna().iteritems():
        tag_list = [tag.strip() for tag in tags.split(",")]
        for tag in tag_list:
            j = labels[tag]
            all_keys[i, j] = 1

    assert np.any(all_keys > 0)
    return all_keys


def repopulate_tags(all_keys, dataset_dense, dataset):
    indices = all_keys.sum(axis=1) > 0
    missing = dataset[np.logical_not(indices)].copy()
    missing["index"] = np.arange(missing.shape[0])

    knn = KNeighborsClassifier(algorithm="auto", n_neighbors=5, weights="distance")
    print("Fitting KNN")
    true_keys = all_keys[indices]
    knn.fit(dataset_dense[indices], true_keys)

    print("Predicting KNN")
    keys_proba = knn.predict_proba(dataset_dense[np.logical_not(indices)])
    print("Done, rearranging")
    for i, val in enumerate(keys_proba):
        temp = val[:, 1] if val.shape[1] == 2 else 1 - val[:, 0]
        keys_proba[i] = temp.reshape((val.shape[0], 1))
    keys_proba = np.hstack(keys_proba)

    repop_keys = np.where(keys_proba > 0, np.ones_like(keys_proba), np.zeros_like(keys_proba))
    assert np.all(repop_keys.sum(axis=1) > 0)
    assert np.all(true_keys.sum(axis=1) > 0)
    for i, ind in missing["index"].iteritems():
        all_keys[i] = repop_keys[ind]

    assert np.all(all_keys.sum(axis=1) > 0)
    return all_keys


def write_files(dataset, dataset_dense, rep_keys):
    N = dataset.shape[0]
    f = h5py.File("dataset.hdf5", "w")
    cos_sims = f.create_dataset("cos_sims", (N, N), compression="gzip")
    key_sims = f.create_dataset("key_sims", (N, N), compression="gzip")
    reliability = f.create_dataset("reliabilities", (N, N), compression="gzip")

    iterations = 10
    n = int(np.ceil(N / iterations))
    lo, hi = 0, n
    i = 0

    lens = [len(text.split(" ")) for _, text in dataset["text"].iteritems()]
    med_len = np.median(lens)
    long_text = np.where(np.array([length > med_len for length in lens]), 1, 0)
    has_desc = np.where(np.array(dataset["company_short_description"].isna()), 0, 1)
    has_keys = np.where(np.array(dataset["company_category_list"].isna()), 0, 1)
    key_counts = np.sum(rep_keys, axis=1).reshape((N, 1))
    big_counts = np.broadcast_to(key_counts, (N, N))
    med_count = np.median(key_counts)

    while lo < N:
        begin = time.time()
        print(f"{i + 1}/{iterations}", end=" ")
        q = hi - lo
        desc = has_desc[lo:hi].reshape((q, 1)) @ has_desc.reshape((1, N))
        keys = has_keys[lo:hi].reshape((q, 1)) @ has_keys.reshape((1, N))
        text = long_text[lo:hi].reshape((q, 1)) @ long_text.reshape((1, N))

        key_lens = (big_counts[lo:hi, :] + big_counts.T[lo:hi, :]) / (2 * med_count)
        reliability[lo:hi, :] = desc * text * (1 + keys * key_lens)

        temp_sims = cosine_similarity(dataset_dense[lo:hi, :], dataset_dense)
        temp_sims = np.where(temp_sims < 0, 0, temp_sims)
        temp_sims = np.where(temp_sims > 1, 1, temp_sims)
        cos_sims[lo:hi, :] = temp_sims

        l, r = big_counts[lo:hi, :], big_counts.T[lo:hi, :]

        key_totals = np.where(l > r, l, r)

        key_sims[lo:hi, :] = (rep_keys[lo:hi, :] @ rep_keys.T) / key_totals
        print(f"took {time.time() - begin} seconds.")

        lo, hi = hi, min(hi + n, N)
        i += 1

    f.close()
