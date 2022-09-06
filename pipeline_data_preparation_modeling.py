from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# for multiple class vars

filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)

array = dataframe.values

X = array[:,0:8]
Y = array[:,8]

#creating piperline
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('lda', LinearDiscriminantAnalysis()))
model = Pipeline(estimators)

num_folds = 10
seed = 7

kfold = KFold(n_splits = num_folds)

results = cross_val_score(model, X, Y, cv=kfold)
print("Mean Estimated Accuracy using Pipeline: %f " % (results.mean()))