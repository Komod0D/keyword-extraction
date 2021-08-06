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


def preprocess(s):
    return re.sub(r"[^a-zA-z0-9\s]", " ", s).lower()


def generate_ngrams(tokens, n=3):
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]


def tokenize(s):
    return [token for token in re.findall(r"\b\w+\b", s)]


files = os.listdir("news")
model = Word2Vec.load("text8.w2v")


print("LOADING TEXTS")
texts = TextDirectoryCorpus("news", character_filters=[preprocess], tokenizer=tokenize,
                            token_filters=[lambda tokens: [t for t in tokens if t in model.wv.vocab]])

print("BUILDING MODEL:")
before = time.time()

print("\tTRAINING MODEL")
model.train(texts.get_texts(), total_examples=len(files), epochs=5)


model.save("with_news.w2v")
print(f"Built in {time.time() - before} seconds")


print(model.vocabulary.raw_vocab)
print(model.wv.most_similar("Artificial Intelligence"))
