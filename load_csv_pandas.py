# use pandas to turn csv into data frame
from pandas import read_csv
filename = 'pima-indians-diabetes.csv'
#could also do a csv url
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
print(data.shape)