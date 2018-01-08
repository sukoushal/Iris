import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('/home/sukoushal/Desktop/Python/Iris/Iris.csv')
ratio = np.random.rand(len(df)) < 0.8

train = df[ratio]
test = df[~ratio]

col = ['Id','SepalLengthCm', 'SepalWidthCm','PetalLengthCm','PetalWidthCm', 'Species']
spe = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = itertools.cycle(["r", "b", "g"])

for i in range(len(spe)):
	y_1 = list(train[train.Species == spe[i]][col[4]])
	x_1 = list(train[train.Species == spe[i]][col[0]])
	plt.scatter(x = x_1, y = y_1,label = col[1])


cat = {'Iris-setosa':1, 'Iris-versicolor':2, 'Iris-virginica':3}
train['cat_map'] = train['Species'].map(cat)
test['cat_map'] = test['Species'].map(cat)

from sklearn.cluster import KMeans
X = train[['SepalWidthCm','SepalLengthCm', 'PetalWidthCm', 'PetalLengthCm']]
y = train['cat_map']

test_df = test[['SepalWidthCm','SepalLengthCm', 'PetalWidthCm', 'PetalLengthCm']]

from sklearn import svm
clf = svm.SVC()
clf.fit(X, y)
predicted = clf.predict(test_df)

#Efficiency'
count = 0

for i in range(len(test_df)):
	if test['cat_map'].iloc[i] == predicted[i]:
		count = count + 1

eff = float(count)/len(test_df)
