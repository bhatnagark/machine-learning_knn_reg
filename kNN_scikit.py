import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import neighbors
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from statistics import mean
import sys

splits = [[100,500,1000,5000,10000,50000,100000,500000,1000000]]

data = pd.read_csv('Sum+noise.csv',sep=';')

target_column = ['Noisy Target']
train_column = ['Feature 1','Feature 2','Feature 3','Feature 4','Feature 6','Feature 7','Feature 8','Feature 9','Feature 10']

for x in splits:
	for i in x:
		X= data[train_column].iloc[:i]
		Y = data[target_column].iloc[:i]

		X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.30,random_state = 42)
		print(i)
		clf = neighbors.KNeighborsRegressor(n_neighbors = 5)
		clf.fit(X_train,Y_train)
		y_pred = clf.predict(X_test)
		mean = np.mean(Y_test)
		mse = np.sqrt(metrics.mean_squared_error(Y_test,y_pred))
		print(mse/mean)
		r2 = metrics.r2_score(Y_test,y_pred)
		print(r2)

sys.modules[__name__].__dict__.clear()

##for dataset Sum without noise

splits = [[100,500,1000,5000,10000,50000,100000,500000,1000000]]

data = pd.read_csv('Sum-noise.csv',sep=';')
print(list(data))
target_column = ['Target']
train_column = ['Feature 1','Feature 2','Feature 3','Feature 4','Feature 6','Feature 7','Feature 8','Feature 9','Feature 10']

for x in splits:
	for i in x:
		X= data[train_column].iloc[:i]
		Y = data[target_column].iloc[:i]

		X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.30,random_state = 42)
		print(i)
		clf = neighbors.KNeighborsRegressor(n_neighbors = 5)
		clf.fit(X_train,Y_train)
		y_pred = clf.predict(X_test)
		mean = np.mean(Y_test)
		mse = np.sqrt(metrics.mean_squared_error(Y_test,y_pred))
		print(mse/mean)
		r2 = metrics.r2_score(Y_test,y_pred)
		print(r2)

sys.modules[__name__].__dict__.clear()

##for dataset Skin_Nonskin

splits = [[100,500,1000,5000,10000,50000,100000]]

data = pd.read_csv('Skin_NonSkin.txt',sep='\t')

target_column = data.iloc[:,3] 
train_column = data.iloc[:,0:2]

for x in splits:
	for i in x:
		X= train_column.iloc[:i]
		Y = target_column.iloc[:i]

		X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.30,random_state = 42)
		print(i)
		clf = neighbors.KNeighborsRegressor(n_neighbors = 5)
		clf.fit(X_train,Y_train)
		y_pred = clf.predict(X_test)
		mean = np.mean(Y_test)
		mse = np.sqrt(metrics.mean_squared_error(Y_test,y_pred))
		print(mse/mean)
		r2 = metrics.r2_score(Y_test,y_pred)
		print(r2)

sys.modules[__name__].__dict__.clear()

##For dataset Year Prediction

splits = [[100,500,1000,5000,10000,50000,100000,500000]]

data = pd.read_csv('YearPredictionMSD.txt',sep=',')

target_column = data.iloc[:,0] 
train_column = data.iloc[:,1:]

for x in splits:
	for i in x:
		X= train_column.iloc[:i]
		Y = target_column.iloc[:i]

		X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.30,random_state = 42)
		print(i)
		clf = neighbors.KNeighborsRegressor(n_neighbors = 5)
		clf.fit(X_train,Y_train)
		y_pred = clf.predict(X_test)
		print(Y_test[0:5])
		print(y_pred[0:5])	
		mean = np.mean(Y_test)
		mse = np.sqrt(metrics.mean_squared_error(Y_test,y_pred))
		print(mse/mean)
		r2 = metrics.r2_score(Y_test,y_pred)
		print(r2)
