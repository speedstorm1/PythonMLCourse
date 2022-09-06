from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)

array = dataframe.values

X = array[:,0:8]
Y = array[:,8]

model = LogisticRegression(solver='liblinear')
rfe = RFE(model, n_features_to_select=3)
fit = rfe.fit(X, Y)

print("Number of features: %d" % fit.n_features_)
print("selected Features : %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)

