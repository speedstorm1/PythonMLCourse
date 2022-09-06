from pickle import load

#name of saved model
filename = 'iris.sav'

#load saved model
loaded_model = load(open(filename, 'rb'))


# define one new data instance for prediction
Xnew = [[6.2,3.1,5.2,2.4]]

# make a prediction
ynew = loaded_model.predict(Xnew)
print("Input =%s, Predicted =%s" % (Xnew[0], ynew[0]))