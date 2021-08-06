import time

from scipy.sparse import coo_matrix, save_npz
import numpy as np
import pandas as pd

new_dataset = pd.read_stata("kmt_kws_june4th2021.dta").head(20000)
new_dataset.reset_index(inplace=True)

all_keywords = [keyword.strip() for _, row in new_dataset['kmt_kws'].iteritems() for keyword in row.split(',')
                if keyword.strip() != '']
all_keywords = set(all_keywords)


important_keywords_dataframe = pd.read_csv('high_weight_keywords.csv')

important_keywords = [keyword.strip() for _, row in important_keywords_dataframe['keywords'].iteritems()
                      for keyword in row.split(';') if keyword.strip() != '']
important_keywords = set(important_keywords)


normal_keywords = all_keywords.difference(important_keywords)

normal_mapping = {keyword: i for i, keyword in enumerate(normal_keywords)}
important_mapping = {keyword: i for i, keyword in enumerate(important_keywords)}

normal_counts = np.zeros(shape=(len(new_dataset), len(normal_keywords)))
important_counts = np.zeros(shape=(len(new_dataset), len(important_keywords)))

print('One hot encoding keywords...', end='\t')
start = time.time()

for i, keywords in new_dataset['kmt_kws'].iteritems():
    for key in keywords.split(','):
        key = key.strip()

        if key in normal_mapping:
            j = normal_mapping[key]
            normal_counts[i, j] = 1

        elif key in important_mapping:
            j = important_mapping[key]
            important_counts[i, j] = 1

print(f'took {time.time() - start:.4f} seconds.')

print('Calculating similarities dense...', end='\t')
start = time.time()

lo, hi = 0, 5000
similarities = normal_counts[lo:hi] @ normal_counts.T + 2 * important_counts[lo:hi] @ important_counts.T

print(f'took {time.time() - start} seconds.')

print('Calculating similarities sparse...', end='\t')
start = time.time()

normal_sparse, important_sparse = coo_matrix(normal_counts).tocsr(), coo_matrix(important_counts).tocsr()

lo, hi = 0, 5000
similarities = normal_sparse[lo:hi] @ normal_sparse.T + 2 * important_sparse[lo:hi] @ important_sparse.T
similarities = similarities.todense()

print(f'took {time.time() - start} seconds.')