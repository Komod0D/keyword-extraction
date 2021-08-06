import h5py
import scipy.sparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def custom_similarity(importances, totals, lo, hi):
    dots = importances[lo:hi] @ importances.T
    sum1, sum2 = totals[lo:hi], totals
    sum1 = sum1[:, np.newaxis]
    sum2 = sum2[np.newaxis, :]
    maxes = np.maximum(sum1, sum2)
    maxes = np.where(maxes == 0, 1, maxes)
    assert maxes.shape == dots.shape

    return dots / maxes


print("Loading data...")
# dataset = pd.read_csv("dataset.csv").reset_index(drop=True)


dataset_sparse = np.load("Custom/word_importances.npy")

# TODO: remove
dataset_sparse = dataset_sparse[:1000]

dataset_sparse = np.where(dataset_sparse > 0, 1, 0)
totals = dataset_sparse.sum(axis=1)


N = 10
n = int(np.ceil(dataset_sparse.shape[0] / N))
lo, hi = 0, n

f = h5py.File("dataset_custom.hdf5", "w")
arr = f.create_dataset("similarities", (dataset_sparse.shape[0], dataset_sparse.shape[0]))

while lo < dataset_sparse.shape[0]:
    print(f"{lo / n} / {N}")
    arr[lo:hi] = custom_similarity(dataset_sparse, totals, lo, hi)
    lo = hi
    hi = min(dataset_sparse.shape[0], hi + n)

f.close()
