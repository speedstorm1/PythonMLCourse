from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

filename = 'BostonHousing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO','B', 'LSTAT', 'MEDV']

dataframe = read_csv(filename, names=names)
array = dataframe.values

X = array[:,0:13]
Y = array[:,13]

kfold = KFold(n_splits=10)
model = LinearRegression()
scoring = 'r2'

results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("r2: %.3f (%.3f)" % (results.mean(), results.std()))