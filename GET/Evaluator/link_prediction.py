import linkpred
import random
from matplotlib import pyplot as plt


emb = input("Select the dataset" )

random.seed(100)

# Read network
G = linkpred.read_network(emb)

# Create test network
test = G.subgraph(random.sample(G.nodes(), 20))

# Exclude test network from learning phase
training = G.copy()
training.remove_edges_from(test.edges())

simrank = linkpred.predictors.SimRank(training, excluded=training.edges())
simrank_results = simrank.predict(c=0.5)

test_set = set(linkpred.evaluation.Pair(u, v) for u, v in test.edges())
evaluation = linkpred.evaluation.EvaluationSheet(simrank_results, test_set)

recall = evaluation.recall()
precision = evaluation.precision()
plt.clf()
plt.plot(recall, precision,  color='navy',
         label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall ')
plt.legend(loc="lower left")
plt.show()