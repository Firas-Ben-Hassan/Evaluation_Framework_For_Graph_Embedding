import numpy as np
import numpy.random as npr


class Graph:
    def __init__(self, graph, is_directed, p, q):
        self.graph = graph
        self.is_directed = is_directed
        self.p = p
        self.q = q
        self.alias_nodes = {}
        self.alias_edges = {}

    def node2vec_walk(self, walk_length, start_node):
        graph = self.graph
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(graph.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prv = walk[-2]
                    walk.append(cur_nbrs[alias_draw(alias_edges[(prv, cur)][0], alias_edges[(prv, cur)][1])])
            else:
                break

        return walk

    def simulate_walks(self, num_walks, walk_length):
        graph = self.graph
        walks = []
        nodes = list(graph.nodes())
        for i in range(num_walks):
            npr.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))
        return walks

    def get_alias_edge(self, src, dst):
        graph = self.graph
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(graph.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(graph[dst][dst_nbr]['weight'] / p)
            elif graph.has_edge(dst_nbr, src):
                unnormalized_probs.append(graph[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(graph[dst][dst_nbr]['weight'] / q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(unn_prob) / norm_const for unn_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        graph = self.graph
        is_directed = self.is_directed

        alias_nodes = {}
        for node in graph.nodes():
            unnormalized_probs = [graph[node][nbr]['weight'] for nbr in sorted(graph.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(unn_prob) / norm_const for unn_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        if is_directed:
            for edge in graph.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in graph.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges


def alias_setup(probs):
    k = len(probs)
    q = np.zeros(k)
    j = np.zeros(k, dtype=np.int)

    # Sort the data into the outcomes with probabilities
    # that are larger and smaller than 1/k.
    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = k * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    # Loop though and create little binary mixtures that
    # appropriately allocate the larger outcomes over the
    # overall uniform mixture.
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        j[small] = large
        q[large] -= (1.0 - q[small])

        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return j, q


def alias_draw(j, q):
    k = len(j)

    # Draw from the overall uniform mixture.
    kk = int(np.floor(npr.rand() * k))

    # Draw from the binary mixture, either keeping the
    # small one, or choosing the associated larger one.
    if npr.rand() < q[kk]:
        return kk
    else:
        return j[kk]