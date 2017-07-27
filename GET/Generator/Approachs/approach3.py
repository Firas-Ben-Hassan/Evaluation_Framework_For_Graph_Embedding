import networkx as nx
from gensim.models import Word2Vec
import node2vec
import numpy.random as npr


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

    edgelist_file = input("Enter graph edgelist filename again : ")
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

    
    return nx_graph


nx_graph = read_graph()   

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

def run():
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

    save_node_features(nm1=node_model1, nm2=node_model2, nodes=nx.nodes(nx_graph), dim=D, output=output)
    print(" LINE ===> Success")
