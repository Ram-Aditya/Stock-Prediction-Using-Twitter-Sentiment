import csv
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import re
from textblob import TextBlob
from collections import defaultdict
import pickle

#Filter out non english sentences
def isEnglish(sentence):
	with open('english_words.txt') as word_file:
		english_words = set(word.strip().lower() for word in word_file)	
	eng_len=0
	for word in sentence:
		correct_word=TextBlob(word.lower()).correct()
		if(correct_word in english_words):
			eng_len+=1
	if((eng_len/len(sentence))<0.8):
		return False
	return True

def parseLargeFile(filename):
	twData=[]
	with open(filename,encoding = "ISO-8859-1") as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		for row in csv_reader:
			# ##print(l)
			if(line_count!=0):
				# sent=row[5]
				# if(file_type=="train" and sent not in ['1','3','5'] ):
					# continue
				tw=row[6]
				if( not isEnglish(tw)):
					continue
				twData.append(sent)
			else:
				line_count=1
	return twData

#Parse csv file to get array
def getArrFromFile(filename,file_type):

	twData=[]
	l=0
	with open(filename,encoding = "ISO-8859-1") as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		for row in csv_reader:
			l+=1
			# ##print(l)
			if(line_count!=0):
				sent=row[5]
				if(file_type=="train" and sent not in ['1','3','5'] ):
					continue
				# tw=row[11]
				# if( not isEnglish(tw)):
				# 	continue
				twData.append(sent)
			else:
				line_count=1
	return twData

# arr=getArrFromFile("apple_tweets.csv","train")
# ##print(len(arr),arr[:10])

#Tokenize each tweet
def tokenizer(dataset):
	tkr = TweetTokenizer(preserve_case=True, reduce_len=False, strip_handles=False)
	tokenized_tw=[]
	for row in dataset:
		tokenized_tw.append(tkr.tokenize(row))
	return tokenized_tw

# tarr=tokenizer(arr[:3])
# ##print(tarr)

#Removing stopwords
def removeStopwords(dataset):
	stopWords = set(stopwords.words('english'))
	count=0
	st_data=[]
	for row in dataset:
		l=[]
		for word_i in range(len(row)):
			if not (row[word_i] in stopWords or row[word_i].lower() in stopWords):
				l.append(row[word_i])
		st_data.append(l)
	# ##print("Count: ",count)
	return st_data

def getSlangDict():
	slang_dict=defaultdict(lambda: None)
	with open("slangdict.txt") as sldict:
		for line in sldict:
			l=[]
			for word in line.split('-'):
				l.append(word.strip().replace('\n','')) 
			# ##print(l)
			if(len(l)==2):
				if(len(l[1].split(" "))>1):
					l[1]=" ".join(word.lower() for word in l[1].split(" "))
				slang_dict[l[0]]=l[1]
	# ##print(len(slang_dict))
	return slang_dict
	
# getSlangDict()
#Regex matching
def removeNoisyTokens(dataset):
	ANY_URL_REGEX = r"""(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))"""
	WEB_URL_REGEX = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""
	IP_REGEX = r"""[0-9]+(?:\.[0-9]+){3}:[0-9]+"""
	user_handle_reg = """(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)"""
	hashtag_reg="""(?<=^|(?<=[^a-zA-Z0-9-_\.]))#([A-Za-z]+[A-Za-z0-9-_]+)"""
	reg1="""(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))""" #Smile Pos
	reg2="""(:\s?D|:-D|x-?D|X-?D)""" #Laugh Pos
	reg3="""(:\s?\(|:-\(|\)\s?:|\)-;)""" #Wink Pos
	reg4="""(<3|:\*)""" #Love Pos
	reg5="""(:\s?\(|:-\(|\)\s?:|\)-:)""" #Sad Neg
	reg6="""(:,\(|:\'\(|:"\()""" #Cry Neg

	slang_dict=getSlangDict()
	cleanData=[]
	c=0
	#Replace URLs, userhandle, hashtags, emoticons, incorrect words, slang words, punctuations
	for row in dataset:
		c+=1
		l=[]
		##print(row)
		for word_i in range(len(row)):
			##print("Word: ",row[word_i],end=" ")
			if len(re.findall(WEB_URL_REGEX,row[word_i]))>0:
				l.append(" URL ")
				##print("URL Case")
				continue
			try:
				if len(re.findall(user_handle_reg,row[word_i]))>0:
					# row[word_i]="USER"
					l.append(" USER ")
					##print("USER")
					continue
			except:
				print(len(re.findall(user_handle_reg,row[word_i])))
				exit()
			hashtag=re.findall(hashtag_reg,row[word_i])
			if len(hashtag)>0 :
				row[word_i]=hashtag[0]
				#print("Hashtag Case")
				# l.append(hashtag[0])

			if(row[word_i].isnumeric()):
				#print("Numeric Case")
				continue

			#Emoticon mapping
			if(len(re.findall(reg1,row[word_i]))>0):
				# row[word_i]=EMO_POS
				l.append(" EMO_POS ")
				#print("Emoti Case")
				continue
			if(len(re.findall(reg2,row[word_i]))>0):
				# row[word_i]=EMO_POS
				l.append(" EMO_POS ")
				#print("Emoti Case")
				continue
			if(len(re.findall(reg3,row[word_i]))>0):
				# row[word_i]=EMO_POS
				l.append(" EMO_POS ")
				#print("Emoti Case")
				continue
			if(len(re.findall(reg4,row[word_i]))>0):
				# row[word_i]=
				l.append(" EMO_POS ")
				#print("Emoti Case")
				continue
			if(len(re.findall(reg5,row[word_i]))>0):
				# row[word_i]=EMO_NEG
				l.append(" EMO_NEG ")
				#print("Emoti Case")
				continue
			if(len(re.findall(reg6,row[word_i]))>0):
				# row[word_i]=EMO_NEG
				l.append(" EMO_NEG ")
				#print("Emoti Case")
				continue

			#Replace prolonged words like cooool with coool and correct typos
			if(row[word_i].isalpha()):
				if(slang_dict[row[word_i].lower()]!=None):
					#print("Slang Case")
					#print("Slang repl",slang_dict[row[word_i]])
					row[word_i]=slang_dict[row[word_i].lower()]
				else:
					#print("Word Case")
					row[word_i]=str(TextBlob(row[word_i]).correct()).lower()
				l.append(row[word_i]+" ")
			else:
				#print("Punct/others Case")
				sentMarkersReg="""(\.+|\?+|!+|;+)"""
				n=re.sub(sentMarkersReg,".",row[word_i])
				if(n=="."):
					#print("New word",n)
					l.append(n)
				# if(row[word_i]!="."):
				# 	del row[word_i]
				# 	break
			#Replace slang words

		sent=" ".join(word.strip() for word in l)
		sent=re.sub("""\.+""",".",sent)
		sents=sent.split(".")

		print("Tweet count: ",c)
		l=""
		for sent in sents:
			if sent.strip()!="":
				l+=sent.strip()
		cleanData.append(l)
		#print("Final ",c,l)

		# cleanData.append(l)

	return cleanData

def preprocess(dataset):

	tk_data=tokenizer(dataset)
	stp_data=removeStopwords(tk_data)
	clean_data=removeNoisyTokens(stp_data)
	# print(clean_data[:9])

	dbfile = open('clean_data_y_3p8k.pkl', 'wb') 
	pickle.dump(clean_data, dbfile) 
	dbfile.close()
	return clean_data

def getModelInput():
	raw_data=getArrFromFile("apple_tweets.csv","train")
	# print(len(raw_data))
	# dbfile = open('clean_data_y_3p8k.pkl', 'wb') 
	# pickle.dump(raw_data, dbfile) 
	# dbfile.close()

	return preprocess(raw_data)
# getModelInput()