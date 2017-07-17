from sklearn.svm import LinearSVC
from gensim.models import doc2vec
from collections import namedtuple
from gensim.models.doc2vec import TaggedDocument
from sklearn.metrics import accuracy_score
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
X = []
Doc_ID =[]
docs = []
label={}
y =[]

def run():
    path = input("Enter dataset path: ")
    with open(path) as f:
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

	print (predicted)
	print("Doc2Vec====> Success")