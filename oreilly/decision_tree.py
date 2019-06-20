import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import os
# tutorial from: https://www.youtube.com/watch?v=XDbj6PxaSf0 
# environment 7:40

PATH = os.path.join('datasets', 'spotify', 'data.csv')
df = pd.read_csv(PATH)

# split dataset
train, test = train_test_split(df, test_size = 0.15)
# print(train.columns)

# build model
c = DecisionTreeClassifier(min_samples_split = 100)

# train the model
features = ['acousticness', 'danceability', 'duration_ms', 'energy',
       'instrumentalness', 'key', 'liveness', 'loudness',
       'speechiness',  'valence']
x_train = train[features]
y_train = train['target']

x_test = train[features]
y_test = train['target']

dt = c.fit(x_train, y_train)

# use model predict
y_pred = c.predict(x_test)
# print(y_pred )

# accuracy
score = accuracy_score(y_test, y_pred)
print(score)