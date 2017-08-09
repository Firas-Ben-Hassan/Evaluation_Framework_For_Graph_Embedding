import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import networkx as nx
from gensim.models import Word2Vec
import node2vec




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

    edgelist_file = input("Enter graph edgelist filename again: ")
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

    
    
      
    
def run():
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
    print(" DeepWalk ===> Success")


        
