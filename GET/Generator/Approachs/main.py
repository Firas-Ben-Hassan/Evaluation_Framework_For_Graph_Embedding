import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import networkx as nx
from gensim.models import Word2Vec
import node2vec
import subprocess


import numpy.random as npr


if __name__ == '__main__':

            from tkinter import *
            fen1 = Tk()

            tex1 = Label(fen1, text='Welcome to The Evaluation Framework of Graph Embeddings for Social Network Analysis' , fg='black')
            tex1.pack()
            bou1 = Button(fen1, text='Start', command = fen1.destroy)
            bou1.pack()
            fen1.mainloop()


            data = {}

            Graph = input ("Is It Edgelist Type            (Boolean form):")
            Label = input ("Is It Label Type               (Boolean form):")
            Text_inf = input ("Is It Text Information Type (Boolean form):")


            data["Graph"] = Graph
            data["Label"] = Label
            data["Text_Information"] = Text_inf


            if ((data["Graph"].lower == "true") and  ( data["Label"].lower()=="false") and ( data["Text_Information"].lower()=="false" )) == True :
                print("According to  your Dataset characteristics, you must select from these graph embeddings approaches")
                print("001) Node2vec")
                print("002) DeepWalk")
                print("003) LINE")
                print("004) DNGR")

            elif ((data["Graph"].lower() =="false") and  ( data["Label"].lower()=="true") and ( data["Text_Information"].lower()=="false"))== True:
                print("According to  your Dataset characteristics, you must select from these graph embeddings approaches")
                print("005) Doc2vec")
                print("006) Paper2vec")

            elif ((data["Graph"].lower() =="False") and  ( data["Label"].lower()=="False ") and ( data["Text_Information"].lower()=="True"))== True :
                print("According to  your Dataset characteristics, you must select from these graph embeddings approaches")
                print("007) Glove")
				print("008) Word2Vec")



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
			elif select == "7" : 
			        import approach7
				emb == approach7.run() 
					
					
					
					
	    print ("your dataset embeddings generation has finished  ") 		
					
					
                    
            
                
        
            
        
            
       
            
            
            
        
        
