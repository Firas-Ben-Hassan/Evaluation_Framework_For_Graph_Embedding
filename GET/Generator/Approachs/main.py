from collections import namedtuple
from gensim.models.doc2vec import TaggedDocument

import numpy as np 
import pandas as pd

  
edgelist = []
Label = []
Text_inf = []
karate = {}

cora_content = {}

arxiv = {}
    
dataset  = input( "Select the dataset :")
    
    # Karate Dataset 
if ( dataset == "karate.edgelist") : 
    f = open('karate.edgelist', 'r') 
    for line in  f.readlines():
            line = line.split()
            edgelist.append([float(x) for x in line[1:]])         
            karate["Edgelist"] = edgelist 
            karate["Label"] = Label
            karate["Text_Information"] = Text_inf
            
            
            
            
            
    # Cora  Dataset        
if ( dataset == "cora.content") : 
    f = open('cora.content', 'r')         

    Doc_ID =[]
    docs = []
    label={}
    y =[]    
    for line in f.readlines():
            line = line.split()
            label[line[0]] = line[-1]
            words=np.array(line[1:-1])
            tags = line[0]
            occ = np.where(words == '1')
            occ = [str(x) for x in occ[0]]
            docs.append(TaggedDocument(occ, [tags]))
            Doc_ID.append(line[0])       
            cora_content["Edgelist"] = edgelist 
            cora_content["Label"] = label
            cora_content["Text_Information"] = Text_inf
    

    
    
    # Karate Dataset 
if ( dataset == "arxiv.edgelist") : 
    f = open('arxiv.edgelist', 'r') 
    for line in  f.readlines():
            line = line.split()
            edgelist.append([float(x) for x in line[1:]])         
            arxiv["Edgelist"] = edgelist 
            arxiv["Label"] = Label
            arxiv["Text_Information"] = Text_inf
        
if (dataset == "arxiv.edgelist" or "karate.edgelist" ) : 

    print("According to  your Dataset characteristics, you must select from these graph embeddings approaches")
    print("001) Node2vec")
    print("002) DeepWalk")
    print("003) LINE")
    print("004) DNGR")

elif ( dataset == "cora.content") :

    print("According to  your Dataset characteristics, you must select from these graph embeddings approaches")
    print("005) Doc2vec")
    print("006) Paper2vec")
    
    

select = input("Enter option: ")
# Node2vec 
if select == "1" :
    import approach1
    emb = approach1.run()
# DeepWalk Approach  
elif select == "2" :
    import approach2
    emb = approach2.run()
# LINE 
elif select == "3" :    
    import approach3
    emb = approach3.run()
# DNGR
elif select == "4" :
    import approach4
    emb = approach4.run()
# Doc2vec
elif select == "5" : 
    import approach5 
    emb = approach5.run()
# Paper2vec
elif select == "6" : 
    import approach6
    emb = approach6.run()
# Glove
print ("your dataset embeddings generation has finished  ") 
        
            
        
            
       
            
            
            
        
        
