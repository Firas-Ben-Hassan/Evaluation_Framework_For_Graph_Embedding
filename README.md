![alt text](https://studip.uni-passau.de/studip/plugins_packages/intelec/PassauOpticsPlugin/assets/images/logo_unipassau.svg)


# Evaluation Framework For Graph Embeddings
                                                        
                                   Deep Neural Networks Based Approaches for Graph Embeddings

                                                   By : Firas BEN HASSAN 
                                                        Supervisors : 
                                                     Mr.Jörg Schlötterer 
                                                     Mrs.Ferdaous CHAABENE
                                                      
                                                 
                                                 
![alt text](http://blog.convergeforimpact.com/wp-content/uploads/2014/11/nln-network-map.png)      

# 1- Abstract 

Graphs, such as social networks, word co-occurrence networks, and communication networks, occur naturally in various real-world applications. Analyzing them yields insight into the structure of society, language, and different patterns of communication. Many approaches have been proposed to perform the analysis.
Recently, methods which use the representation of graph nodes in vector space have gained traction from the research community.


# 2-Tasks 

Node Classiﬁcation :

 Often in networks, a fraction of nodes are labeled. In social networks, labels may indicate interests, beliefs, or demographics. In language networks, a document may be labeled with topics or keywords, whereas the labels of entities in biology networks may be based on functionality. Due to various factors, labels may be unknown for large fractions of nodes. For example, in social networks many users do not provide their demographic information due to privacy concerns. Missing labels can be inferred using the labeled nodes and the links in the network. The task of predicting these missing labels is also known as node classiﬁcation. 
 
Link Prediction :

Networks are constructed from the observed interactions between entities, which may be incomplete or inaccurate. The challenge often lies in identifying spurious interactions and predicting missing information. Link prediction refers to the task of predicting either missing interactions or links that may appear in the future in an evolving network. 

 
 # 3-Datasets 
 
 Social Networks Datasets : 
  
-KARATE : 

Zachary’s karate network is awell-known social network of a university karate club
Social network of friendships between 34 members of a karate club at a US university in the 1970 

-BLOGCATALOG :

 This is a network of social relationships of the bloggers listed on the BlogCatalog website. The labels represent blogger interests inferred through the metadata provided by the bloggers.
 The network has 10,312 nodes, 333,983 edges and 39 different labels.
 
-LiveJournal:

LiveJournal is a free on-line blogging community where users declare friendship each other. LiveJournal also allows users form a group which other members can then join. We consider such user-defined groups as ground-truth communities. We provide the LiveJournal friendship social network and ground-truth communities. 

| Name        | Type      | Nodes | Edges  | Link |
| ------------- |:-------------:| --------- |:----------:| -----:|
| **Karate**     | Undirected,Unweighted,Static  | 34|78| [click here to download](https://github.com/tkipf/kerasgcn/tree/master/kegra/data/cora) |
| **BlogCatalog**     | Undirected,Unweighted   |   10312 |333983 | [click here to download](http://www.blogcatalog.com)  |
| **LiveJournal** | Undirected,Unweighted  | 3 997 962  |34681189  | [click here to download](http://snap.stanford.edu/data/com-LiveJournal.html ) |


Collaboration Networks Datasets : 

-Cora:

The Cora dataset consists of Machine Learning papers
 The papers were selected in a way such that in the final corpus every paper cites or is cited by at least one other paper. There are 2708 papers in the whole corpus. 
 
-Wiki:

Wiki contains 2, 405 documents from 19 classes and 17, 981 links between them. 

-Citeseer:

Citeseer contains 3, 312 publications from six classes and 4, 732 links between them. Similar to Cora, the links are citation relationships between the documents and each paper is described by a binary vector of 3, 703 dimensions. 

| Name        | Type      | Nodes | Edges  | Link |
| ------------- |:-------------:| --------- |:----------:| -----:|
| **Cora**     | Undirected,Unweighted,  | 2708 |5429 | [click here to download](www.research.whizbang.com/data ) |
| **Wiki**     | Undirected,Unweighted   |  2405   | 17981 | [click here to download](https://github.com/albertyang33/TADW )  |
| **Citeseer** | Undirected,Unweighted  | 3312  |4732  | [click here to download](https://github.com/albertyang33/TADW) |

Biology  Networks Dataset : 


-PROTEIN-PROTEIN INTERACTIONS (PPI) : 

This is a network of biological interactions between proteins in humans. This network has 3,890 nodes and 38,739 edges.

| Name        | Type      | Nodes | Edges  | Link |
| ------------- |:-------------:| --------- |:----------:| -----:|
| **PPI**     | Undirected,Unweighted,  |3890 |38739 | [click here to download](https://thebiogrid.org/) |


                                                                 
# 4- Paper References with the implementation(s) and the dataset(s)

**Node2vec**

[node2vec (Network-only): Scalable Feature Learning for Networks](http://dl.acm.org/citation.cfm?id=2939672.2939754), 

[[arxiv]](https://arxiv.org/abs/1607.00653) 
[[Python]](https://github.com/aditya-grover/node2vec)
[[Python]](https://github.com/apple2373/node2vec) 
[[Python]](https://github.com/PFE-Passau/Graph_Embeddings),

datasets(Cora,  Zachary’s Karate Club,  BlogCatalog,  Wikipedia,  PPI)



**DeepWalk**

[DeepWalk (Network-only): Online Learning of Social Representations](http://dl.acm.org/citation.cfm?id=2623732),

[[arxiv]](https://arxiv.org/abs/1403.6652) 
[[Python]](https://github.com/phanein/deepwalk)
[[Python]](https://github.com/PFE-Passau/Graph_Embeddings),

datasets(Cora,  Zachary’s Karate Club,  BlogCatalog,  Wikipedia,  PPI)



**LINE**

[LINE(Network-only): Large-scale information network embedding](http://dl.acm.org/citation.cfm?id=2741093), 

[[arxiv]](https://arxiv.org/abs/1503.03578)
[[C++]](https://github.com/tangjianpku/LINE)
[[Python]](https://github.com/PFE-Passau/Graph_Embeddings),

datasets(Cora,  Zachary’s Karate Club,  BlogCatalog,  Wikipedia,  PPI)

**Doc2vec**
[Doc2vec (content-only ): Distributed Representations of Sentences and Document](http://dl.acm.org/citation.cfm?id=3053062&CFID=772667669&CFTOKEN=64514719),

[[arxiv]](https://arxiv.org/abs/1607.05368)
[[Python]](https://github.com/PFE-Passau/Doc2Vec),

dataset (Cora)

**Paper2vec**
[Paper2vec ( combined ): Combining Graph and Text Information for Scientiﬁc Paper Representation](https://researchweb.iiit.ac.in/~soumyajit.ganguly/papers/P2v_1.pdf),


[[arxiv]](https://arxiv.org/abs/1703.06587)
[[Python]](https://github.com/asxzy/paper2vec-gensim),

 dataset (Cora)

**Glove**
[Glove (content-only ): global vectors for word representation](http://dl.acm.org/citation.cfm?id=2889381&CFID=772667669&CFTOKEN=64514719),

[[Python]](https://github.com/jroakes/glove-to-word2vec)


**GraRep**

[Grarep: Learning graph representations with global structural information](http://dl.acm.org/citation.cfm?id=2806512), 

[[Matlab]](https://github.com/ShelsonCao/GraRep),
[[Datasets]](https://github.com/ShelsonCao/GraRep/blob/master/code/core/GetCoOccMatFromGraph.m)


**TADW**

[TADW ( combined ): Network Representation Learning with Rich Text Information](http://dl.acm.org/citation.cfm?id=2832542), 

[[paper]](https://www.ijcai.org/Proceedings/15/Papers/299.pdf) 
[[Matlab]](https://github.com/thunlp/tadw),

Datasets (Cora,  Citeseer,  Wikipedia)


**planetoid**

[Planetoid: (Network-only)]Revisiting Semi-supervised Learning with Graph Embeddings, 

[[arxiv]](https://arxiv.org/abs/1603.08861) 
[[Python]](https://github.com/kimiyoung/planetoid),

Datasets (Cora,  Citeseer,  Wikipedia)


**DNGR**

[DNGR: (Network-only) Deep Neural Networks for Learning Graph Representations](http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12423),

[[Matlab]](https://github.com/ShelsonCao/DNGR)
[[Python Keras]](https://github.com/MdAsifKhan/DNGR-Keras),
[[Datasets]](https://github.com/MdAsifKhan/DNGR-Keras/blob/master/wine_network.mat)

**ComplEx**
[ComplEx :(Network-only)Complex Embeddings for Simple Link Prediction](http://dl.acm.org/citation.cfm?id=3045609),

[[arxiv]](https://arxiv.org/abs/1606.06357) 
[[Python]](https://github.com/ttrouill/complex),
[[Datasets]](https://github.com/ttrouill/complex/tree/master/datasets)


# 5- Evaluation Framework for Graph Embeddings Approaches 

![alt text](http://www.kennedypearce.com/wp-content/uploads/2015/08/strengths-weaknesses-sized.jpg)    

The rationale behind our framework is to provide developers, end users and researchers with easy-to-use interfaces that allow for the agile, ﬁne-grained and uniform evaluation of graph embeddings appraoches on multiple tasks and on multiple datasets.

By these means, we aim to ensure that both tool developers and end users can derive meaningful insights pertaining to the extension, integration and use of graph embeddings algorithms.


In particular, the evaluation framework  provides comparable results to tool developers so as to allow them to easily discover the strengths and weaknesses of their implementations with respect to the state of the art and it allows deriving insights pertaining to the areas in which tools should be further reﬁned, thus allowing developers to create an informed agenda for extensions and end users to detect the right tools for their purposes.

The evaluation framework  is an open-source and extensible framework that allows evaluating tools against (currently) 11 diﬀerent approaches on 2 diﬀerent tasks with 7 different  datasets.






