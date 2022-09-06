from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from pickle import dump

filename = 'iris.data.csv'
names = ['sepal_length','sepal_width','petal_length','petal_width','class']

dataset = read_csv(filename, names=names)
array = dataset.values

X = array[:,0:4]
Y = array[:,4]

# check class distribution to make sure data is even: print(dataset.groupby('class').size())
model = LogisticRegression(multi_class='multinomial', solver='newton-cg')
model.fit(X, Y)

filename = 'iris.sav'
dump(model, open(filename, 'wb'))