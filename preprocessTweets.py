import csv
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

#Parse csv file to get array
def getArrFromFile(filename):

	twData=[]
	with open(filename,encoding = "ISO-8859-1") as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		for row in csv_reader:
			if(line_count!=0):
				sent=row[5]
				if(sent not in ['1','3','5']):
					continue
				tw=row[11]
				twData.append([tw,sent])
			else:
				line_count=1
	return twData

#Tokenize each tweet
def tokenizer(dataset):
	tkr = TweetTokenizer(preserve_case=True, reduce_len=False, strip_handles=False)
	tokenized_tw=[]
	for row in dataset:
		tokenized_tw.append([tkr.tokenize(row[0]),row[1]])
	return tokenized_tw

#Removing stopwords
def removeStopwords(dataset):
	stopWords = set(stopwords.words('english'))
	tw_data=[]
	for row in dataset:
		for word in row[0]:
			if word in stopWords:
				row[0].remove(word)
		tw_data.append([row[0],row[1]])
	return tw_data

arr=getArrFromFile('apple_tweets.csv')
arr=tokenizer(arr)
print(len(arr),arr[0])