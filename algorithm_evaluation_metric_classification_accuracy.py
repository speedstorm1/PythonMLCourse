from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)

#k fold with num folds = num data points

array = dataframe.values

X = array[:,0:8]
Y = array[:,8]


kfold = KFold(n_splits=10)
model = LogisticRegression(solver='liblinear')
scoring = 'accuracy' # - for correct or not


results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Accuracy: %.3f%% (%.3f%%) " % (results.mean()*100.0, results.std()*100.0))