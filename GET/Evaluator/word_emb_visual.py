import numpy as Math
import pylab as Plot

emb = input ("Select the file of embeddings ") 
label = input("Select the file of labels ") 

matrix = Math.loadtxt(emb);
words = [line.strip() for line in open(label)]

target_words = [line.strip().lower() for line in open(label)][:2000]

rows = [label.index(word) for word in target_words if word in label]
target_matrix = matrix[rows,:]
reduced_matrix = tsne(target_matrix, 2);
Plot.figure(figsize=(200, 200), dpi=100)
max_x = Math.amax(reduced_matrix, axis=0)[0]
max_y = Math.amax(reduced_matrix, axis=0)[1]
Plot.xlim((-max_x,max_x))
Plot.ylim((-max_y,max_y))

Plot.scatter(reduced_matrix[:, 0], reduced_matrix[:, 1], 20);

for row_id in range(0, len(rows)):
    target_word = label[rows[row_id]]
    x = reduced_matrix[row_id, 0]
    y = reduced_matrix[row_id, 1]
    Plot.annotate(target_word, (x,y))

Plot.savefig("embeddings.png")