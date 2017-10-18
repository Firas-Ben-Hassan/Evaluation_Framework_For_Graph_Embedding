print ("Welcome to GEVI")

print ("1- Check our Dataset's market in GEVI")
print ("2- Check our graph embedding approaches's market in GEVI")
print ("3- Generate graph embedding using new dataset")
print ("4- Generate graph embedding using existing dataset")
print ("5- Generate graph embedding using new approah")
option = input("Enter option: ")
if (option == "1"): 
    print("1) Collaboration networks datasets")
    print("2) Social networks datasets")  
    print("3) Biology Networks datasets")       
    type1 = input("Enter option: ")  
    if (type1 == "1"):       
        from PIL import Image
        bidule = Image.open("1.png")
        bidule.show()
    if (type1 == "2"):       
        from PIL import Image
        bidule = Image.open("2.png")
        bidule.show() 
    if (type1 == "3"):       
        from PIL import Image
        bidule = Image.open("3.png")
        bidule.show()
if (option == "2"):     
    from PIL import Image
    bidule = Image.open("4.png")
    bidule.show() 
    
if (option == "3"):     
    print ("If you want to upload a new dataset , please select its type (Edgelist, Label or Text information):")
    print("1) Edgelist")
    print("2) Label")
    print("3) Text Information")
    type_dataset = input("Enter option: ")

    if (type_dataset == "1"): 
        print("According to  your Dataset characteristics, you must select from these graph embeddings approaches")
        print("1) Node2vec")
        print("2) DeepWalk")
        print("3) LINE")

    elif ( type_dataset =="2" ) :
        print("According to  your Dataset characteristics, you must select from these graph embeddings approaches")
        print("4) Doc2vec")
        print("5) Paper2vec")

    elif ( type_dataset== "3" ) :
        print("According to  your Dataset characteristics, you must select from these graph embeddings approaches")
        print("6) Word2vec")
        print("7) Glove")    


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

if (option == "4"): 
    print ("1- For node embedding, GEVI offers Karate, Cora, Arxiv  ") 
    print ("2- For Doc embedding, GEVI offers Cora.content  ") 
    print ("3- For word embedding, GEVI offers questions-words  ") 
    select = input ("Select option ")
    if select == "1" :
        print("You must select from these graph embeddings approaches")
        print("1) Node2vec")
        print("2) DeepWalk")
        print("3) LINE")
    if select == "2" :
        print("You must select from these graph embeddings approaches")
        print("4) Doc2vec")
        print("5) Paper2vec")
    if select == "3" :
        print("You must select from these graph embeddings approaches")
        print("6) Word2vec")
        print("7) Glove") 
    sel = input("Enter option: ")   
    if sel == "1" :
            import n2v
            emb = n2v.run()
        # DeepWalk   
    elif sel == "2" :
            import deepwalk
            emb = deepwalk.run()
        # LINE 
    elif sel == "3" :    
            import line
            emb = line.run()
        # Doc2vec
    elif sel == "4" : 
            import  doc2vec
            emb = doc2vec.run()
        # Paper2vec
    elif sel == "5" : 
            import paper2vec
        # Word2vec    
    elif sel =="6":  
            import word2vec
        # Glove
    elif sel =="7":
            import glove 
    print ("your dataset embeddings generation has finished  ")  
if (option == "5"): 
    print ("1- If your new approach corresponds to learning continuous feature representations for nodes in networks, GEVI offers Karate, Cora, Arxiv  as datasets") 
    print ("2- If your new approach corresponds to learning continuous feature representations for documents, GEVI offers Cora.content  ") 
    print ("3- If your new approach corresponds to learning continuous feature representations for words, GEVI offers questions-words  ") 
    select = input ("Select option ")
