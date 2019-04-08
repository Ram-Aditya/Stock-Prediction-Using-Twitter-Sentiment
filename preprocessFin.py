import csv
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import re
from textblob import TextBlob
from collections import defaultdict
import pickle
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import pandas as pd
from SVM import SVM
import random
import sklearn.metrics as metrics

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

import matplotlib.pyplot as plt


def getArrFromFile():

	twData=[]
	l=0
	ign=[]
	with open("financial_data_AAPL.csv") as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		data=[]
		data2=[]
		l=0
		arr=[]
		l=0
		for row in csv_reader:
			if(line_count!=0):
				if((l-2)%5==0):
					arr.append("Unavailable")
					arr.append("Unavailable")
				arr.append(row)
			else:
				line_count=1
			l+=1
		csv_reader=arr
		# print("hghf",len(arr))
		# l=line_count=0
		for row_i in range(len(csv_reader)):
			if(row_i==0):
				data2.append("Unavailable")
				continue
			try:
				close2=int(float(csv_reader[row_i][2]))
			except:
				data2.append("Unavailable")
				continue
			try:
				close1=int(float(csv_reader[row_i-1][3]))
			except:
				data2.append("Unavailable")
				continue
			if (close2-close1)>0:
				data2.append(1)
			else:
				data2.append(0)

	count=0
	for row in data2:
		if row!="Unavailable":
			count+=1
	# print("Grand count",count)
	with open("twitter_data_AAPL.csv") as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		data=[]
		for row in csv_reader:
			if(line_count!=0):
				neg_tw=int(float(row[1]))
				neu_tw=int(float(row[2]))
				pos_tw=int(float(row[3]))
				data.append([neg_tw,neu_tw,pos_tw])
			else:
				line_count=1
	print(len(data),len(data2))
	# for i in range(len(data)):
	# 	data[i].append(data2[i])
	return data2,data

def makeFeatures(fin_data,tw_data):
	features=[]
	for i in range(3,len(fin_data)):
		if(fin_data[i]!="Unavailable"):
			for j in range(i-1,i-4,-1):
				neg=neu=pos=0
				if(j>=0):
					neg+=tw_data[j][0]
					neu+=tw_data[j][1]
					pos+=tw_data[j][2]
					features.append([neg,neu,pos,fin_data[i]])
	print(features[0:10])
	random.seed(10)
	random.shuffle(features)
	return features

def train(features):
	tf =pd.DataFrame(features)
	tf.sample(frac=1,random_state=9)
	print("Df dimension",tf.shape)
	print(tf.iloc[:,36:])
	X = tf.iloc[0:468,0:3]
	X=preprocessing.scale(X)
	Y = tf.iloc[0:468,3]
	X_test = tf.iloc[500:,0:3]
	X_test=preprocessing.scale(X_test)
	Y_test = tf.iloc[500:,3]
	parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
	# svc = SVC(gamma="auto",verbose=False)
	# clf = GridSearchCV(svc, parameters, cv=5) 
	# clf.fit(X,Y)
	# y_pred = clf.predict(X_test)
	# print("Accuracy is ",clf.score(X_test,Y_test))


	lr=LogisticRegression()
	lr.fit(X,Y)
	y_pred=lr.predict(X_test)
	acc=accuracy_score(Y_test,y_pred)
	print("Accuracy: ",acc)


	fpr, tpr, threshold = metrics.roc_curve(Y_test, y_pred)
	roc_auc = metrics.auc(fpr, tpr)
	print(precision_recall_fscore_support(y_true, y_pred, average='macro'))

	# # method I: plt
	# plt.title('Receiver Operating Characteristic')
	# plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
	# plt.legend(loc = 'lower right')
	# plt.plot([0, 1], [0, 1],'r--')
	# plt.xlim([0, 1])
	# plt.ylim([0, 1])
	# plt.ylabel('True Positive Rate')
	# plt.xlabel('False Positive Rate')
	# plt.show()

	# X = tf.iloc[0:300,0:3]
	# # X=preprocessing.scale(X)
	# Y = tf.iloc[0:300,3]
	# X_test = tf.iloc[500:,0:3]
	# # X_test=preprocessing.scale(X_test)
	# Y_test = tf.iloc[500:,3]
	# model = SVM(max_iter=10000, kernel_type='linear', C=1.0, epsilon=0.001)
	# support_vectors, iterations =model.fit(X, Y)
	# sv_count = support_vectors.shape[0]
	# y_pred = model.predict(X_test)

	# acc = calc_acc(Y_test, y_pred)
	# print(acc)

	# print("Support vector count: %d" % (sv_count))
	# print("bias:\t\t%.3f" % (model.b))
	# print("w:\t\t" + str(model.w))
	# print("accuracy:\t%.3f" % (acc))
	# print("Converged after %d iterations" % (iterations))

	# parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
	# svc = SVC(gamma="auto",verbose=False)
	# clf = GridSearchCV(svc, parameters, cv=5) 
	# clf.fit(X,Y)
	# y_pred = clf.predict(X_test)

	# print("Predicted output is ",y_pred)


fin_data,tw_data=getArrFromFile()
ft=makeFeatures(fin_data,tw_data)
train(ft)
