import networkx as nx
import matplotlib.pyplot as plt #

dataset  = input("Enter the dataset: ")
#lecture d'une liste de liens
G=nx.read_edgelist(dataset, create_using=nx.DiGraph())

#visualisation
pos=nx.spring_layout(G)               #choix de l'algorithme
nx.draw(G, pos, with_labels=True)
plt.show()
