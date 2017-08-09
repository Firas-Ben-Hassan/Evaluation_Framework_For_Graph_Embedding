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

dataset = input("Select the dataset")
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
    
# classify with 50% data as training dataset
X = []
Y = []
with open('doc2vec.embd','w') as f:
    f.write("%s %s\n"%(len(datasets),100))
    for y,key in enumerate(labels.keys()):
        for index,paper in enumerate(labels[key]):
            f.write(paper+" "+" ".join([str(x) for x in model.docvecs[paper]])+"\n")
            X.append(model.docvecs[paper])
            Y.append(y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)
clf = SVC(kernel='rbf',C=1.5).fit(X_train,y_train)


# classify with 10-fold
parameters = {
    "kernel":["rbf"],
    "C" :[1.5]
             }
tunedclf = GridSearchCV(clf,parameters,cv=10,n_jobs=24)
tunedclf.fit(X,Y)
G = defaultdict(dict)

for data in datasets:
    for n in model.docvecs.most_similar(data.tags,topn=2):
        G[data.tags[0]][n[0]] = None
        G[n[0]][data.tags[0]] = None

with open('cora.content') as f:
    for line in f:
        line = line.rstrip().split("\t")
        try:
            G[line[0]][line[1]] = None
            G[line[1]][line[0]] = None
        except:
            print(line)

neighbors = []

# default parameters for deepwalk
# 10 iterations
for i in range(10):
    for node in G:
        path = [node]
        # 40 walks per node
        while len(path) < 40:
            cur = path[-1]
            path.append(random.choice(list(G[cur].keys())))
        neighbors.append(path)
        
from gensim.models.word2vec import Word2Vec
p2v = Word2Vec(size=100, window=5, min_count=0)
p2v.build_vocab(neighbors)
p2v.intersect_word2vec_format('doc2vec.embd')

