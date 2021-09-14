#!/usr/bin/python3

import random
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from outlier_cleaner import outlierCleaner



### load up some practice data with outliers in it --> list
ages = joblib.load( open("practice_outliers_ages.pkl", "rb") )
net_worths = joblib.load( open("practice_outliers_net_worths.pkl", "rb") )

### ages and net_worths need to be reshaped into 2D numpy arrays
### second argument of reshape command is a tuple of integers: (n_rows, n_columns)
### by convention, n_rows is the number of data points
### and n_columns is the number of features

ages       = np.reshape( np.array(ages), (len(ages), 1))
net_worths = np.reshape( np.array(net_worths), (len(net_worths), 1))


ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(
    ages, net_worths, test_size=0.1, random_state=42)


reg = LinearRegression().fit(ages_train, net_worths_train)

print(f"model is fitted with {reg.coef_[0]} unit slope and {reg.intercept_} intercept")

train_score = reg.score(ages_train,net_worths_train)
test_score = reg.score(ages_test,net_worths_test)

print(f"model completed with training score of {train_score:0.3f} and test score of {test_score:0.3f}")

try:
    plt.plot(ages, reg.predict(ages), color="blue")
except NameError:
    pass
plt.scatter(ages, net_worths)
plt.show()

### identify and remove the most outlier-y points
cleaned_data = []
try:
    predictions = reg.predict(ages_train)
    cleaned_data = outlierCleaner( predictions, ages_train, net_worths_train )
except NameError:
    print("Your regression object doesn't exist, or isn't name reg")
    print("Can't make predictions to use in identifying outliers")


print(f"model completed with training score of {train_score:0.3f} and test score of {test_score:0.3f}")


print(len(cleaned_data))
print('-'*50)


### only run this code if cleaned_data is returning data
if len(cleaned_data) > 0:
    ages, net_worths, errors = zip(*cleaned_data)
    ages       = np.reshape( np.array(ages), (len(ages), 1))
    net_worths = np.reshape( np.array(net_worths), (len(net_worths), 1))

    ### refit your cleaned data!
    try:
        
        reg.fit(ages, net_worths)

        cleaned_score = reg.score(ages,net_worths)

        print(f"on cleaned trained data the score was {cleaned_score:0.3f}")
        print("-"*70)

        cleand_test_score = reg.score(ages_test,net_worths_test)
        print(f"the score of the new model on test data: {cleand_test_score:0.3f}")
        


        plt.plot(ages, reg.predict(ages), color="blue")


    except NameError:
        print("You don't seem to have regression imported/created,")
        print("   or else your regression object isn't named reg")
        print("   either way, only draw the scatter plot of the cleaned data")
    plt.scatter(ages, net_worths)
    plt.xlabel("ages")
    plt.ylabel("net worths")
    plt.show()


else:
    print("outlierCleaner() is returning an empty list, no refitting to be done")


