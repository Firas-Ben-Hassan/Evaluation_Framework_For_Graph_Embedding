from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np 
import pandas as pd
from sklearn.metrics import f1_score

class run() : 
    data=[]
    label={}
    y =[]

    with open("C:/Users/asus/Documents/GitHub/Evaluation_Framework_For_Graph_Embeddings/GET/Evaluator/embeddings") as f:
        for line in  f.readlines():
            line = line.split()
            data.append([float(x) for x in line[1:]])

    with open("C:/Users/asus/Documents/GitHub/Evaluation_Framework_For_Graph_Embeddings/GET/Generator/Datasets/cora.content") as f:
        for line in f.readlines(): 
            line = line.split()
            label[int(line [0])] = line[-1]

    with open("C:/Users/asus/Documents/GitHub/Evaluation_Framework_For_Graph_Embeddings/GET/Evaluator/embeddings") as f:
        for line in f.readlines():
            line = line.split()
            y.append(label[int(line[0])])

    X_train, X_test, y_train , y_test = train_test_split( data, y,test_size=0.8, random_state=0)

    #train_test_split?


    clf =LinearSVC()
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    micro = f1_score(y_test, predicted, average='micro')
    macro = f1_score(y_test, predicted, average='macro')
    
    accuracy_score = accuracy_score(y_test, predicted)
    print (micro , macro, accuracy_score)
