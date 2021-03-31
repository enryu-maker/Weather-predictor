import pandas
from matplotlib import pyplot 
import sklearn
from sklearn import linear_model
import numpy as np



dataset = pandas.read_csv('data.csv')
dataset.drop('timestamp', inplace=True, axis=1)
#dataset.hist()
#pyplot.show()
#print(dataset)
predict = 'Temperature'
x = np.array(dataset.drop([predict], 1))
y = np.array(dataset[predict])

x_train, x_test, y_train, y_test= sklearn.model_selection.train_test_split(x,y, test_size = 0.1)
linear= linear_model.LinearRegression()

linear.fit(x_train,y_train)
acc= linear.score(x_test,y_test)
print(acc*100)
print('co',linear.coef_)
print('intercept',linear.intercept_)
value=[]
humi=int(input('Humidity==>'))
value.append(humi)
speed=int(input('speed==>'))
value.append(speed)
direction=int(input('direction==>'))
value.append(direction)
predication= linear.predict([value])

print(predication)