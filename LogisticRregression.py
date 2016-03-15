print(__doc__)


import numpy as np
from sklearn import linear_model, cross_validation
import csv
from sklearn.cross_validation import KFold
from sklearn.decomposition import PCA

data_training = np.genfromtxt('/Users/nesbtesh/Downloads/HW4.training.csv', delimiter=",")
data_test = np.genfromtxt('/Users/nesbtesh/Downloads/HW4.test.csv', delimiter=",")

#get X values
#ask if I only have to pick the first 7
X_train = data_training[:, 0:22]
X_test = data_test[:, 0:22]

#get Y variables
Y_train = data_training[:, 22]
#Y_test = data_test[:, 21]

# Create Logistic regression object
regr = linear_model.LogisticRegression()

# separate the train data into differentes samples
kf = KFold(len(X_train), n_folds=2)

#inite score list
score_model1 = []

#itirate the samples
for train_index, test_index in kf:

	#fit the train data
	regr.fit(X_train[train_index], Y_train[train_index])

	#Score with test data that we got from the kfold
	score_model1.append(regr.score(X_train[test_index], Y_train[test_index]))

#print the scores 
print reduce(lambda x, y: x + y, score_model1) / len(score_model1)

#generate prediction with the actual test data
prediction = regr.predict(X_test)

# #open file and write the results into a csv file
# myfile = open('results2.csv', 'wb')
# wr = csv.writer(myfile, dialect='excel')
# #wr.writerows(prediction)
# for row in prediction:
# 	wr.writerow([row])


#######################################With PCA#########################
#perform PCA
pca = PCA(n_components=22)
pca.fit(X_train)
print(pca.explained_variance_ratio_)

#from the variance ratio we choose the first 15 variables 
pca.n_components = 15
X_reduced = pca.fit_transform(X_train)
X_reduced.shape

# separate the train data into differentes samples
kf = KFold(len(X_reduced), n_folds=2)

score_model1 = []

#itirate the samples
for train_index, test_index in kf:

	#fit the train data
	regr.fit(X_reduced[train_index], Y_train[train_index])

	#Score with test data that we got from the kfold
	score_model1.append(regr.score(X_reduced[test_index], Y_train[test_index]))

#print the scores 
print reduce(lambda x, y: x + y, score_model1) / len(score_model1)

#######################################With PCA2#########################
#perform PCA
pca = PCA(n_components=22)
pca.fit(X_train)
print(pca.explained_variance_ratio_)

pca.n_components = 12
X_reduced = pca.fit_transform(X_train)
X_reduced.shape

# separate the train data into differentes samples
kf = KFold(len(X_reduced), n_folds=2)

score_model1 = []

#itirate the samples
for train_index, test_index in kf:

	#fit the train data
	regr.fit(X_reduced[train_index], Y_train[train_index])

	#Score with test data that we got from the kfold
	score_model1.append(regr.score(X_reduced[test_index], Y_train[test_index]))

#print the scores 
print reduce(lambda x, y: x + y, score_model1) / len(score_model1)

#we train the model with the choosen variables
X_train = data_training[:, 0:12]
regr.fit(X_train[train_index], Y_train[train_index])

#generate prediction with the actual test data
X_test = data_test[:, 0:12]
prediction = regr.predict(X_test)

#open file and write the results into a csv file
myfile = open('results2.csv', 'wb')
wr = csv.writer(myfile, dialect='excel')
#wr.writerows(prediction)
for row in prediction:
	wr.writerow([row])
