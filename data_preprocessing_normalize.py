from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import Normalizer
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)


array = dataframe.values
# seperate into input and output components
X = array[:,0:8]
Y = array[:,8]

scaler = Normalizer().fit(X)
rescaledX = scaler.transform(X)

set_printoptions(precision=3)
print(rescaledX[0:5,:])