from gensim.models import Word2Vec 
import nltk
from tqdm import tqdm
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
dbfile=open("clean_data.pkl","rb")
clean_data=pickle.load(dbfile)
dataset=[]
count=0
indarr=[]
for row in clean_data:
	l=row.strip().split(" ")
	if(len(l)>2):
		dataset.append(l)
		indarr.append(count)
	count+=1
print(len(dataset))
# print(dataset[0:10])s
# print(cdata[100:110])
# print(cl)
# model=Word2Vec()
# model.build_vocab([x for x in dataset])
# model.train(dataset,total_examples=model.corpus_count,epochs = model.iter)
# print(model["pains"])

x_train = dataset
tweet_w2v = Word2Vec(size=100, min_count=1)
tweet_w2v.build_vocab([x for x in tqdm(x_train)])
tweet_w2v.train([x for x in tqdm(x_train)],total_examples=tweet_w2v.corpus_count,epochs=tweet_w2v.iter)
print(tweet_w2v["gave"])