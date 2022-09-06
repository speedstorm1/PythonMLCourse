from pandas import read_csv
from matplotlib import pyplot
import numpy
import scipy
from pandas import set_option
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
from pandas.plotting import scatter_matrix

# univariate histograms
data.hist()
pyplot.show()

# density plot - histogram with curve over top
data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
pyplot.show()

# box and whisker plot
data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
pyplot.show()



# multivariate plots

# correlation color matrix
correlations = data.corr()
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()

# scatter plot matrix
scatter_matrix(data)
pyplot.show()