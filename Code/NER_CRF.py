#  #######
#  # ### #
#  ### # #
#      # #
#      ###        Udacity Machine Learning Nanodegree Capstone Project
#                 Drafted 4/01/2018.
#      ###        Nitin Bhandari

import pandas as pd
import numpy as np
from sklearn_crfsuite import CRF
from sklearn.cross_validation import cross_val_predict
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.metrics import confusion_matrix
import eli5
from sklearn.metrics import confusion_matrix


# Importing the dataset
data = pd.read_csv("ner_dataset.csv", encoding="latin1")
# Filling in Null Values
data = data.fillna(method='ffill')

# Priniting last 10 rows of the DataFrame
print('Data Peek={}'.format(data.tail(10)))
print('---------------------------------')

words = list(set(data['Word'].values))
n_words = len(words)

def get_sentence(data):
    data = data
    fxn = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
    grouped = data.groupby("Sentence #").apply(fxn)
    sentences = [s for s in grouped]
    return sentences

sentences = get_sentence(data)

print('---------------------------------')
print('This is how a sentence looks like={}'.format(sentences[0][0]))

# Feature Extraction
def word_to_features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent_to_features(sent):
    return [word_to_features(sent, i) for i in range(len(sent))]

def sent_to_labels(sent):
    return [label for token, postag, label in sent]

def sent_to_tokens(sent):
    return [token for token, postag, label in sent]

X = [sent_to_features(s) for s in sentences]
y = [sent_to_labels(s) for s in sentences]

crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=20,
          all_possible_transitions=False)

pred = cross_val_predict(estimator=crf, X=X, y=y, cv=5)

report = flat_classification_report(y_pred=pred, y_true=y)
print(report)

crf.fit(X, y)

crf = CRF(algorithm='lbfgs', c1=0.2, c2=0.2, max_iterations=20,
          all_possible_transitions=False)

pred = cross_val_predict(estimator=crf, X=X, y=y, cv=5)

report = flat_classification_report(y_pred=pred, y_true=y)
print(report)
confusion_matrix = confusion_matrix(y, pred)
print("Confusion matrix:\n%s" % confusion_matrix)

crf.fit(X, y)
eli5.show_weights(crf, top=30)
