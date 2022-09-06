# Grid Search for Algorithm Tuning
from scipy.stats import uniform

import numpy
from pandas import read_csv
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV

filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]


param_grid = {'alpha': uniform()}
model = Ridge()
grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=3, n_iter=100)
grid.fit(X, Y)
print(grid.best_score_)
print(grid.best_estimator_.alpha)