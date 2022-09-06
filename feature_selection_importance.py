from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier

filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)

array = dataframe.values

#splitting the array to input and output
X = array[:,0:8]
Y = array[:,8]

#feature selection using feature importance method
model = ExtraTreesClassifier(n_estimators=100)
model.fit(X, Y)
print(model.feature_importances_)


