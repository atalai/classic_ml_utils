#   @@@@@@@@@@@@@@@@@@@@@@@@
#   **Code by Aron Talai****
#   @@@@@@@@@@@@@@@@@@@@@@@@

# Classification utils

#def libs
import os
import scipy 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from data_cleaning import drop_all_nan 
from pre_process import zero_mean_unit_std


# read data
data = pd.read_csv(os.getcwd() + '/SAheart.csv')
data = data.drop(columns = ['IDS'])

# global init 
dont_normalized_list = ['age','chd']


# data cleaning
data = drop_all_nan(data)

# remove every outlier based on +- 2std away form the mean
# also normalize the data
for row in data.columns:
	try:
		mean = data[row].mean()
		std = data[row].std()
		data = data[(data[row] < mean + 2*std) & (data[row] > mean - 2*std)]

		if row not in dont_normalized_list:
			data[row]=(data[row]-mean)/std

	#pass on categorical cells
	except:
		pass 


# data.to_csv(os.getcwd() + '/test1.csv')
# convert categorical data to numerical i.e. famhist
data['famhist'] = data['famhist'].astype("category").cat.codes

##find inter-feature corrolations
# chd is only modestly corrolated with age and tobacco usage.
# no collinearity in the dataset
#print (data.corr(method = 'pearson')) 

# convert dataframe to numpy and prepare for classifcation
data = data.values
features = data[:, 0:int(data.shape[1]-1)]
labels = data[:,-1]

# establish train/test (90/10) split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, shuffle = True)

# declare classifiers 
#random forest
rf_clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=1993)
rf_model = rf_clf.fit(X_train, y_train)  
rf_acc = rf_model.score(X_test,y_test)
print (rf_acc)

# lda
ld_clf = LinearDiscriminantAnalysis(solver='eigen')
ld_model = ld_clf.fit(X_train, y_train)  
ld_acc = ld_model.score(X_test,y_test)
print (ld_acc)

#svm
svm_clf = LinearSVC()
svm_model = svm_clf.fit(X_train, y_train)
svm_acc = svm_model.score(X_test,y_test)
print (svm_acc)

# logistic regression
lr_clf = LogisticRegression(solver='lbfgs', multi_class='ovr')
lr_model = lr_clf.fit(X_train, y_train)  
lr_acc = lr_model.score(X_test,y_test)
print (lr_acc)

#naive bayes
gnb_clf = GaussianNB()
gnb_model = gnb_clf.fit(X_train, y_train)  
gnb_acc = gnb_model.score(X_test,y_test)
print (gnb_acc)

conf_mat = confusion_matrix(y_test, gnb_model.predict(X_test))
print(conf_mat)