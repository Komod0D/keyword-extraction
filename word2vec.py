import re
import pandas as pd
import gensim.downloader as api
import numpy as np
from gensim.models import Word2Vec
from gensim.corpora.textcorpus import *
import os
from functools import partial
import h5py
from nltk import PorterStemmer
import time

"""
industries = pd.read_csv("industries.csv")
industries = set([ind.strip().lower() for _, group in industries["Industries"].iteritems() for ind in group.split(",")
                  if ind.strip().lower() != ""])

token_pattern = re.compile('(?u)\\b[\\w-]+\\b')


for ind in industries:
    if not token_pattern.fullmatch(ind):
        print(ind)
        print(token_pattern.findall(ind))
        print()
"""


def preprocess(s):
    return re.sub(r'[^a-zA-Z-_\s]', ' ', s.title())


def generate_ngrams(tokens, n=3):
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]


def tokenize(s):
    token_pattern = re.compile('(?u)\\b[\\w-]+\\b')
    tokens = re.findall(token_pattern, s)
    trigrams = generate_ngrams(tokens, 3)
    bigrams = generate_ngrams(tokens, 2)

    return tokens + bigrams + trigrams


files = os.listdir("news")

print("LOADING TEXTS")
texts = TextDirectoryCorpus("news", character_filters=[preprocess], tokenizer=tokenize, token_filters=[])

print("BUILDING MODEL:")
before = time.time()
model = Word2Vec(workers=4)

print("\tBUILDING VOCAB")
model.build_vocab(sentences=texts.get_texts(), progress_per=1000)

print("\tTRAINING MODEL")
model.train(texts.get_texts(), total_examples=len(files), epochs=5)


model.save("word2vec.mdl")
print(f"Built in {time.time() - before} seconds")


print(model.vocabulary.raw_vocab)
print(model.wv.most_similar("Artificial Intelligence"))
