from pandas import read_csv
from pandas import set_option
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

# print first 20 rows
peek = data.head(20)
print(peek)

# print data types of dataframe
types = data.dtypes
print(types)

# descriptive statistics
set_option('display.width', 100)
set_option('precision', 3)
description = data.describe()
print(description)

# class distribution - for classification problems
class_counts = data.groupby('class').size()
print(class_counts)

# correlation ranging from -1 to 1
set_option('display.width', 100)
set_option('precision', 3)
correlations = data.corr(method='pearson')
print(correlations)

# skewness
# positive skewed is where the peak is to the left of the bell curve (median < mean)
skew = data.skew()
print(skew)
