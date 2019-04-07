from gensim.models import Word2Vec 
import nltk
# from tqdm import tqdm
from preprocessTweets import *
# nltk.download('brown')
# # nltk.download('movie_reviews')
# # nltk.download('treebank')
# # nltk.download('punkt')
# from nltk.corpus import brown#, movie_reviews, treebank
# b = Word2Vec(brown.sents())

# print(b.most_similar('money',topn=5))

# model=Word2Vec(getModelInput())
dbfile=open("clean_data","rb")
clean_data=pickle.load(dbfile)
dataset=[]
for row in clean_data:
	l=row.strip().split(" ")
	if(len(l)>2):
		dataset.append(l)
print(dataset[0:10])
# print(cdata[100:110])
# print(cl)
model=Word2Vec()
model.build_vocab([x for x in dataset])
model.train(dataset,total_examples=model.corpus_count,epochs = model.iter)
print(model["pains"])
