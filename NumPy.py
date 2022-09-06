# defining numpy array
import numpy
mylist = [1, 2, 3]
myarray = numpy.array(mylist)
print(myarray)
print(myarray.shape)

mylist = [[1, 2, 3], [3, 4, 5]]
myarray = numpy.array(mylist)
print(myarray)
print(myarray.shape)
print("First row: %s" % myarray[0,2])
print("Last row: %s" % myarray[-1])
print("Specific row and col: %s" % myarray[0,2])
print("Whole col: %s" % myarray[:,2])

# numpy arithmetic
myarray1 = numpy.array([2, 2, 2])
myarray2 = numpy.array([3, 3, 3])
print("addition %s" % (myarray1 + myarray2))
print("multiplication %s" % (myarray1 * myarray2))
