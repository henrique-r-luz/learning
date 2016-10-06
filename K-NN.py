import os
import csv
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data_filename = os.path.join("data", "tic-tac-toe.csv")
X = np.zeros((958, 9), dtype='float')
y = np.zeros((958,), dtype='str')

#X = np.array()
#y = np.array()


'''with open(data_filename, 'r') as input_file:
    reader = csv.reader(input_file)
    for i, row in enumerate(reader):
        # Get the data, converting each item to a float
        data = [float(datum) for datum in row[:-1]]
        # Set the appropriate row in our dataset
        #print(data)
        X[i] = data
        # 1 if the class is 'g', 0 otherwise
        y[i] = row[-1] == 'g'''


with open(data_filename, 'r') as input_file:
    reader = csv.reader(input_file)
    for i, row in enumerate(reader):
        # Get the data, converting each item to a float
        item = []
        for datum in row[:-1]:
            if(datum=='x'):
                datum = 0
            if(datum=='o'):
                datum = 1
            if(datum=='b'):
                datum = 0.5
            item.append(datum)
        # Set the appropriate row in our dataset
        print(item)
        X[i] = item
        # 1 if the class is 'g', 0 otherwise
        y[i] = row[-1] == 'positive'

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=14)
#print("There are {} samples in the training dataset".format(X_train.shape[0]))
#print("There are {} samples in the testing dataset".format(X_test.shape[0]))
#print("Each sample has {} features".format(X_train.shape[1]))

estimator = KNeighborsClassifier()
estimator.fit(X_train, y_train)
y_predicted = estimator.predict(X_test)
accuracy = np.mean(y_test == y_predicted) * 100
print("The accuracy is {0:.1f}%".format(accuracy))

