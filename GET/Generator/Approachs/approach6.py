import nltk
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
import networkx as nx
from collections import defaultdict,namedtuple,Counter
from glob import glob
import sys
import os
import math
import random
from six.moves import xrange
if sys.version_info[0] >= 3:
    unicode = str

import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,train_test_split
from gensim.models.word2vec import Word2Vec

random.seed(0)
np.random.seed(0)

CORA = namedtuple('CORA', 'words tags')

datasets = []
labels = defaultdict(list)
with open("cora.content") as f:
    for line in f:
        line = line.split()
        ID = line[0]
        labels[line[-1]].append(ID)
        words = []
        for i,w in enumerate(line[1:-1]):
            if w == "1":
                words.append(str(i))
        datasets.append(
            CORA(
                words,
                [ID]
            )
        )

logging.info("done... %s papers loaded" % (len(datasets)))
logging.info("%s labels" % (len(labels)))

import random
from gensim.models.doc2vec import Doc2Vec
#model = Doc2Vec(dbow_words=1,iter=5,batch_words=100,negative=20,min_count=0,sample=0.001,dm=0)
model = Doc2Vec(alpha=0.025, window=10, min_count=10, min_alpha=0.025, size=100)
model.build_vocab(datasets)

# decrease alpha
for i in range(10):
    random.shuffle(datasets)
    model.alpha = 0.025-0.002*i
    model.min_alpha = model.alpha
    model.train(datasets)

p2v = Word2Vec(size=100, window=5, min_count=0)
p2v.build_vocab(neighbors)
p2v.intersect_word2vec_format('doc2vec.embd')
p2v.train(neighbors)
