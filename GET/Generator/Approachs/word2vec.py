from gensim.models import word2vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
dataset = input ("Select your dataset")
sentences = word2vec.Text8Corpus(dataset)
model = word2vec.Word2Vec(sentences, size=200)
model.save("embeddings.model")