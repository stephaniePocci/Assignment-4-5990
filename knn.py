#-------------------------------------------------------------------------
# AUTHOR: Stephanie Pocci
# FILENAME: knn.py
# SPECIFICATION: 
# FOR: CS 5990- Assignment #4
# TIME SPENT: 
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd

#11 classes after discretization
bins = [i for i in range(-22, 40, 6)]
bins.append(40) # ensure upper bound is included

#defining the hyperparameter values of KNN
neighbors = [k for k in range(1, 20)]
power_values = [1, 2]
weight_options = ['uniform', 'distance']

#reading the training data
#reading the test data
#hint: to convert values to float while reading them -> np.array(df.values)[:,-1].astype('f')
train_df = pd.read_csv('weather_training.csv', sep=',', header=0)
train_array = np.array(train_df.values)
train_df['Temperature (C)'] = pd.cut(train_df['Temperature (C)'], bins=bins, labels=False, right=False)

X_train = np.array(train_df.values)[:, 1:-1]
y_train_raw = np.array(train_df.values)[:, -1:]
y_train = [elem for sub in y_train_raw for elem in sub]

test_df = pd.read_csv('weather_test.csv', sep=',', header=0)
test_array = np.array(test_df.values)
test_df['Temperature (C)'] = pd.cut(train_df['Temperature (C)'], bins=bins, labels=False, right=False)

X_test = np.array(train_df.values)[:, 1:-1]
y_test_raw = np.array(train_df.values)[:, -1:]
y_test = [elem for sub in y_test_raw for elem in sub]

#loop over the hyperparameter values (k, p, and w) of KNN
#--> add your Python code here

max_accuracy = -1
opt_k = -1
opt_p = -1
opt_w = ''

for n in neighbors:
    for p_val in power_values:
        for w_type in weight_options:

            #fitting the knn to the data
            #--> add your Python code here

            model = KNeighborsClassifier(n_neighbors=n, p=p_val, weights=w_type)
            model.fit(X_train, y_train)

            #make the KNN prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously, use zip()
            #Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
            #to make a prediction do: clf.predict([x_testSample])
            #the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values.
            #to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
            #--> add your Python code here

            hits = 0 # number of correct predictions
            predictions = model.predict(X_test)

            for idx in range(len(predictions)):
                real_output = y_test[idx]
                guess = predictions[idx]

                print("pred:", real_output)
                print("real:", guess)

                if guess == 0:
                    percent_error = 0
                else:
                    percent_error = abs(100 * (abs(real_output - guess) / guess))

                if percent_error <= 15:
                    hits += 1

            #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
            #with the KNN hyperparameters. Example: "Highest KNN accuracy so far: 0.92, Parameters: k=1, p=2, w= 'uniform'"
            #--> add your Python code here

            current_accuracy = hits / len(predictions)

            if current_accuracy > max_accuracy:
                max_accuracy = current_accuracy
                opt_k = n
                opt_p = p_val
                opt_w = w_type
                print(f"Highest KNN accuracy so far: {max_accuracy}")
                print(f"Parameters: k={n}")
                print(f"Parameters: p={p_val}")
                print(f"Parameters: w={w_type}")

print("---------------------")
print("END TESTING")
print("---------------------")
print(f"Highest KNN accuracy: {max_accuracy}")
print(f"Parameters: k={opt_k}")
print(f"Parameters: p={opt_p}")
print(f"Parameters: w={opt_w}")
