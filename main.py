import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import networkx as nx
from gensim.models import Word2Vec
import node2vec


import numpy.random as npr


if __name__ == '__main__':
    
       
        Approaches_n = input("Number of Approaches that you would like to evaluate them ")
        i = 0
        while i < int(Approaches_n) : 
            print("Select Algorithm to train")
            print("1) Node2vec")
            print("2) DeepWalk")
            print("3) Doc2vec")
            print("4) Glove")
            print("5) Planetoid")
            print("6) DNGR")
            print("7) ComplEx")
            select = input("Enter option: ")

            # Node2vec Approach
            if select == "1" :
                import approach1
                emb = approach1.run()
            # DeepWalk Approach  
            elif select == "2" :
                import approach2
                emb = approach2.run()
            # Doc2vec 
            elif select == "3" :
                import approach3
                emb = approach3.run()
            # Glove 
            elif select == "4" : 
                import approach4
                emb = approach4.run()
            # Planetoid
            #elif select == "5"  : 
                #import approach5
                #emb = approah5.run()
            
            
        
            
        
            
       
            
            
            
        
        
