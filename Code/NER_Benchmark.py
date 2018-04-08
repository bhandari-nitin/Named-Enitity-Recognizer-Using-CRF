#  #######
#  # ### #
#  ### # #
#      # #
#      ###        Udacity Machine Learning Nanodegree Capstone Project
#     ####        Benchmark Model using Preceptron classifier (Linear Model)
#                 Drafted 4/01/2018.
#      ###        Nitin Bhandari
##
## Importing required libraries
import pandas as pd
import numpy as np
from sklearn.cross_validation import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
# reading data using pandas
# pandas returns the DataFrame
data = pd.read_csv("ner_dataset.csv", encoding = "latin1")
# Filling in null values
data = data.fillna(method='ffill')
# Taking a peek into the dataset
print data.head(10)

words = list(set(data['Word'].values))
n_words = len(words)

words = data["Word"].values.tolist()
tags = data["Tag"].values.tolist()

# Creating a simple feature map
def feature_map(word):
    return np.array([word.istitle(), word.islower(), word.isupper(),
                    len(word), word.isdigit(),  word.isalpha()])

words = [feature_map(w) for w in words]

clf = Perceptron(n_jobs=-1, n_iter=5)
y_predict = cross_val_predict(clf, X=words, y=tags, cv=5, verbose=10)
report = classification_report(y_pred=y_predict, y_true=tags)
print(report)
