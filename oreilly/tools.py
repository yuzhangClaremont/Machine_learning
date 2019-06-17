import os
import tarfile
import urllib
import numpy as np
import pandas as pd



def fetch_data(data_url, data_path):
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    tgz_path = os.path.join(data_path, "housing.tgz")
    urllib.request.urlretrieve(data_url, tgz_path) # download from url and copyt to the path
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=data_path)
    housing_tgz.close()


# For illustration only. Sklearn has train_test_split()
def split_train_test(data, test_ratio, data_path):
    shuffled_indices = np.random.permutation(len(data)) # a shuffled list of integer
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    train = data.iloc[train_indices]
    test = data.iloc[test_indices]
    train_path = os.path.join(data_path, 'train_data.csv')
    test_path = os.path.join(data_path, 'test_data.csv')
    train.to_csv(train_path)
    test.to_csv(test_path)

# train, test = split_train_test(housing_data, 0.2)
# print(test.describe())
# print(train.describe())