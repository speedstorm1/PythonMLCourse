from pandas import read_csv
from numpy import set_printoptions
from sklearn.decomposition import PCA

filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)

array = dataframe.values

X = array[:,0:8]
Y = array[:,8]

pca = PCA(n_components=3)
fit = pca.fit(X)

print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)