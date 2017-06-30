import networkx as nx
from gensim.models import Word2Vec
import node2vec
import numpy.random as npr
from sklearn.svm import LinearSVC
from gensim.models import doc2vec
from collections import namedtuple
from gensim.models.doc2vec import TaggedDocument


def discard_edges(nx_graph, p):
    if p == 0:
        return nx_graph
    else:
        ebunch = nx_graph.edges()
        npr.shuffle(ebunch)
        for i in range(0, len(ebunch), p):
            nx_graph.remove_edges_from([ebunch[i]])
        return nx_graph


def read_graph():
    print(" *******Welcome to the evaluation framework for graph embeddings algorithms *****")
    print(" Enter the type of your Dataset: 1--> Edgelist , 2--> Label , 3--> Text Information ")
    edgelist_file = input("Enter graph edgelist filename: ")
    is_unweighted = input("Unweighted graph (Y/N): ")
    is_undirected = input("Undirected graph (Y/N): ")

    if is_unweighted == "Y":
        nx_graph = nx.read_edgelist(edgelist_file,  create_using=nx.DiGraph())
        for edge in nx_graph.edges():
            nx_graph[edge[0]][edge[1]]['weight'] = 1
    else:
        nx_graph = nx.read_edgelist(edgelist_file,  data=(('weight', float), ), create_using=nx.DiGraph())

    if is_undirected == "Y":
        nx_graph = nx_graph.to_undirected()

    # P = int(input("Enter fraction of edges to discard (0-9): "))
    # return discard_edges(nx_graph, P)
    return nx_graph


def learn_node_features(walks, dim, window, epoch, output):

    emb_walks = [[str(n) for n in walk] for walk in walks]
    node_model = Word2Vec(emb_walks, size=dim, window=window, min_count=0, sg=1, workers=4, iter=epoch)
    node_model.wv.save_word2vec_format(output)


def learn_node_features_2(walks, dim, window, epoch):

    emb_walks = [[str(n) for n in walk] for walk in walks]
    node_model = Word2Vec(emb_walks, size=dim, window=window, min_count=0, sg=1, workers=4, iter=epoch)
    return node_model


def save_node_features(nm1, nm2, nodes, dim, output):

    with open(output, 'w') as out:
        fv = [str(len(nodes)) + " " + str(dim) + "\n"]
        for n in nodes:
            nr = [n] + list(nm1[str(n)]) + list(nm2[str(n)])
            fv.append(" ".join([str(r) for r in nr]) + "\n")
        out.writelines(fv)

if __name__ == '__main__':
    nx_graph = read_graph()
    Approaches_n = input("Number of Approaches that you would like to evaluate them ")
    i = 1
    while i < int(Approaches_n) :
        print("Select Algorithm to train")
        print("1) Node2vec")
        print("2) DeepWalk")
        print("3) LINE")
        print("4) Doc2vec")
        print("5) Paper2vec")
        print("6) Glove")
        print("7) DNGR")
        print("8) Planetoid")
        select = input("Enter option: ")
        
        # Node2vec Approach

        if select == "1":
            print("Based on previous experiments the best in-out and return hyperparameters are {0.25, 0.50, 1, 2, 4}")
            P = float(input("Enter in-out parameter: "))
            Q = float(input("Enter return parameter: "))
            graph = node2vec.Graph(nx_graph, is_directed=nx.is_directed(nx_graph), p=P, q=Q)
            graph.preprocess_transition_probs()
            num_walks = int(input("Enter no. of walks to sample for each node: "))
            walk_length = int(input("Enter length of each walk: "))
            walks = graph.simulate_walks(num_walks=num_walks, walk_length=walk_length)

            D = int(input("Enter dimensionality of the feature vectors: "))
            W = int(input("Enter window size: "))
            epoch = int(input("Enter number of iterations: "))
            output = input("Enter output file: ")
            learn_node_features(walks=walks, dim=D, window=W, epoch=epoch, output=output)
            
            # Deep Walk Approach

        elif select == "2":
            P = 1
            Q = 1
            graph = node2vec.Graph(nx_graph, is_directed=nx.is_directed(nx_graph), p=P, q=Q)
            graph.preprocess_transition_probs()
            num_walks = int(input("Enter no. of walks to sample for each node: "))
            walk_length = int(input("Enter length of each walk: "))
            walks = graph.simulate_walks(num_walks=num_walks, walk_length=walk_length)

            D = int(input("Enter dimensionality of the feature vectors: "))
            W = int(input("Enter window size: "))
            epoch = int(input("Enter number of iterations: "))
            output = input("Enter output file: ")
            learn_node_features(walks=walks, dim=D, window=W, epoch=epoch, output=output)

        # LINE Approach 
            
        elif select == "3":
            num_walks = int(input("Enter no. of walks to sample for each node: "))
            walk_length = int(input("Enter length of each walk: "))
            D = int(input("Enter dimensionality of the feature vectors: "))
            W = int(input("Enter window size: "))
            epoch = int(input("Enter number of iterations: "))
            output = input("Enter output file: ")

            P = 0.001
            Q = 1
            graph = node2vec.Graph(nx_graph, is_directed=nx.is_directed(nx_graph), p=P, q=Q)
            graph.preprocess_transition_probs()
            walks = graph.simulate_walks(num_walks=num_walks, walk_length=walk_length)
            node_model1 = learn_node_features_2(walks=walks, dim=D/2, window=W, epoch=epoch)

            P = 1
            Q = 0.001
            graph = node2vec.Graph(nx_graph, is_directed=nx.is_directed(nx_graph), p=P, q=Q)
            graph.preprocess_transition_probs()
            walks = graph.simulate_walks(num_walks=num_walks, walk_length=walk_length)
            node_model2 = learn_node_features_2(walks=walks, dim=D/2, window=W, epoch=epoch)

        # Doc2vec   

        elif select == "4":
            Doc_ID =[]
            docs = []
            y =[]

            for line in  edgelist_file.readlines():
                line = line.split()
                label[line[0]] = line[-1]
                words=np.array(line[1:-1])
                tags = line[0]
                occ = np.where(words == '1')
                occ = [str(x) for x in occ[0]]
                docs.append(TaggedDocument(occ, [tags]))
                Doc_ID.append(line[0])  
         
            model = doc2vec.Doc2Vec(docs, size = 100, dm = 0,  window = 5 , min_count = 1, workers = 4, iter = 10, negative = 10)       
        
            for document_id in Doc_ID:
                X.append(model.docvecs[document_id])
                y.append(label[document_id]) 

            print(model)
            
            # 
            
            save_node_features(nm1=node_model1, nm2=node_model2, nodes=nx.nodes(nx_graph), dim=D, output=output)
