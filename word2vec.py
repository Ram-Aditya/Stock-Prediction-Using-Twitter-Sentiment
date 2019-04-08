from gensim.models import Word2Vec 
import nltk
from tqdm import tqdm
from preprocessTweets import *

dbfile=open("clean_data.pkl","rb")
clean_data=pickle.load(dbfile)
print(len(clean_data))

dataset=[]
count=0
indarr=[]
for row in clean_data:
	l=row.strip().split(" ")
	if(len(l)>2):
		dataset.append(l)
		indarr.append(count)
	count+=1

print(dataset[:10])

x_train = dataset
tweet_w2v = Word2Vec(size=250, min_count=3)
tweet_w2v.build_vocab([x for x in tqdm(x_train)])
tweet_w2v.train([x for x in tqdm(x_train)],total_examples=tweet_w2v.corpus_count,epochs=tweet_w2v.iter)
print(tweet_w2v["pain"])