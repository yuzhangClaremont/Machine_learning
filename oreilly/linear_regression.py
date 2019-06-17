
# Code example
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
import tools
import os

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

tools.fetch_data(HOUSING_URL, HOUSING_PATH)

DATA_PATH = os.path.join(HOUSING_PATH,'housing.csv')
housing_data = pd.read_csv(DATA_PATH)
# print(housing_data.head())

tools.split_train_test(housing_data, 0.2, HOUSING_PATH)

TRAIN_PATH = os.path.join(HOUSING_PATH,'train_data.csv')

train_data = pd.read_csv(TRAIN_PATH)

corr_matrix = train_data.corr()
print(corr_matrix)


# Prepare the data
X_median_income = np.c_[train_data["median_income"]]
Y_total_rooms = np.c_[train_data["total_rooms"]]


print(train_data.head())

# Visualize the data
train_data.plot(kind='scatter', x="median_income", y="total_rooms", alpha = 0.1) # alpha: transparency
plt.show()

# # Select a linear model
model = linear_model.LinearRegression()

# # Train the model
model.fit(X_median_income, Y_total_rooms)

# # Make a prediction for Cyprus
X_new = [[4]]  
print(model.predict(X_new))