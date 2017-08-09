from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from gensim.models import doc2vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np 
import pandas as pd



data=[]
label={}
y =[]

task = input("Select the task ")
print("1) Node Classification ")
print("2) Label Classificaion")


if (task == "1") :  
             
    emb = input ("Select the file of embeddings ") 
    label = input("Select the file of labels ") 
    
    with open(emb) as f:
        for line in  f.readlines():
            line = line.split()
            data.append([float(x) for x in line[1:]])

    with open(label) as f:
        for line in f.readlines(): 
            line = line.split()
            label[int(line [0])] = line[-1]

    with open(emb) as f:
        for line in f.readlines():
            line = line.split()
            y.append(label[int(line[0])])

    X_train, X_test, y_train , y_test = train_test_split( data, y,test_size=0.8, random_state=0)

    #train_test_split?


    clf =LinearSVC()
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)

    fichier = open("result.txt", "w")
    fichier.write("The acuracy is :" )
    fichier.write(str( accuracy_score(y_test, predicted)))
    fichier.write("\n")
    fichier.write("Micro F1 is :" )
    fichier.write(str(f1_score(y_test, predicted, average='micro')))
    fichier.write("\n")
    fichier.write("Maco F1 is :" )
    fichier.write(str(f1_score(y_test, predicted, average='macro')))
    fichier.close()
             
             
elif task == "2":
             
    emb = input ("Select the  embedding's file ") 
    label = input("Select the labels file ") 
    X = []
    Doc_ID =[]
    docs = []
    label={}
    y =[]


    with open(label) as f:

        for line in f.readlines():
            line = line.split()
            label[line[0]] = line[-1]
            words=np.array(line[1:-1])
            tags = line[0]
            occ = np.where(words == '1')
            occ = [str(x) for x in occ[0]]
            docs.append(TaggedDocument(occ, [tags]))
            Doc_ID.append(line[0])  


    # Building the model for Doc2Vec   

    model = doc2vec.Doc2Vec(docs, size = 100, dm = 0,  window = 5 , min_count = 1, workers = 4, iter = 10, negative = 10)       
    # Getting the embeddings 
    for document_id in Doc_ID:
        X.append(model.docvecs[document_id])
        y.append(label[document_id])    


    #print(model.docvecs[2707])       


    #    SVM Classifier                  

    X_train, X_test, y_train , y_test = train_test_split( X, y,test_size=0.7, random_state=0)


    clf =LinearSVC()
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    fichier = open("result.txt", "w")
    fichier.write("The acuracy is :" )
    fichier.write(str( accuracy_score(y_test, predicted)))
    fichier.write("\n")
    fichier.write("Micro F1 is :" )
    fichier.write(str(f1_score(y_test, predicted, average='micro')))
    fichier.write("\n")
    fichier.write("Maco F1 is :" )
    fichier.write(str(f1_score(y_test, predicted, average='macro')))
    fichier.close() 
