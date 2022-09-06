import numpy
import pandas

# series - 1d array where you can label rows and cols
myarray = numpy.array([1,2,3])
rownames = ['a', 'b', 'c']
myseries = pandas.Series(myarray, index=rownames)
print(myseries)
print(myseries[0])
print(myseries['a'])

#dataframe - 2d array with labels
myarray1 = numpy.array([[1,2,3], [4,5,6]])
rownames1 = ['a', 'b']
colnames1 = ['one', 'two', 'three']
mydataframe = pandas.DataFrame(myarray1, index=rownames1, columns=colnames1)
print(mydataframe)

print("one column: %s" % mydataframe['one'])
print("one column: %s" % mydataframe.one)