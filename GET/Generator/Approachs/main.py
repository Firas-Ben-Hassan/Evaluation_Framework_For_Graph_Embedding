type_dataset = input ("Select the type of the dataset (Edgelist, Label or Text information):")

if (type_dataset == "edgelist"): 
    print("According to  your Dataset characteristics, you must select from these graph embeddings approaches")
    print("001) Node2vec")
    print("002) DeepWalk")
    print("003) LINE")

elif ( type_dataset =="label" ) :
    print("According to  your Dataset characteristics, you must select from these graph embeddings approaches")
    print("004) Doc2vec")
    print("005) Paper2vec")
          
          
elif ( type_dataset== "text" ) :
    print("According to  your Dataset characteristics, you must select from these graph embeddings approaches")
    print("006) Word2vec")
    print("007) Glove")    
    

select = input("Enter option: ")
# Node2vec 
if select == "1" :
    import n2v
    emb = n2v.run()
# DeepWalk   
elif select == "2" :
    import deepwalk
    emb = deepwalk.run()
# LINE 
elif select == "3" :    
    import line
    emb = line.run()
# Doc2vec
elif select == "4" : 
    import  doc2vec
    emb = doc2vec.run()
# Paper2vec
elif select == "5" : 
    import paper2vec
# Word2vec    
elif select =="6":  
    import word2vec
# Glove
elif select =="7":
    import glove 
print ("your dataset embeddings generation has finished  ") 

 