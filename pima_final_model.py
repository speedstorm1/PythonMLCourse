from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from pickle import dump

filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

dataframe = read_csv(filename, names=names)
array = dataframe.values

X = array[:,0:8]
Y = array[:,8]

# should test different models to optimize accuracy

model = LogisticRegression(solver='liblinear')
model.fit(X, Y)

filename = 'final_pima_indian.sav'
dump(model,open(filename, 'wb'))