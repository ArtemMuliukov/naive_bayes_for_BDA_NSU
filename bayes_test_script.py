import numpy as np
import pandas as pd
import arff #liac-arff
from collections import defaultdict, Counter
import naive_bayes
import warnings
warnings.filterwarnings("ignore")


def cross_val_scorer(clf, X, y, n = 3 ,random_state = None):

    if random_state is not None:
        np.random.seed(random_state)
    if n > X.shape[0]:
        raise Exception("n must be less of number of elements in X")
    step = X.shape[0] // n
    perm = np.random.permutation(X.shape[0])
    X = X.iloc[perm]
    y = y[perm]
    scores = np.zeros(n)
    for i in range(n):
        c = np.array(range(X.shape[0]))
        mask_in = (c >= i * step) & (c < (i + 1) * step)
        mask_out = (c < i * step) | (c >= (i + 1) * step)
        clf.fit(X.iloc[mask_out],y[mask_out])
        preds = clf.predict(X.iloc[mask_in])
        #print(preds.values())
        scores[i] = len(y[mask_in][y[mask_in] == preds])/len(y[mask_in])
    return scores


def run_test(test_name = 'test1', r_s = 228, num_of_splits = 5):

    if test_name != 'test1' and test_name != 'test2':
        raise Exception("Test name should be test1 or test2")

    if test_name == 'test2':
        dataset_big = arff.load(open('soybean.arff', 'r', encoding="utf-8"))
        attrs = dataset_big['attributes']
        attrs = [x[0] for x in attrs]
        data = pd.DataFrame(dataset_big['data'],columns = attrs)
        data = data.dropna(axis=0, how='any')
        data = data.reindex(index=range(data.shape[0]))
        X = data.drop(["class"], axis=1)
        y = data['class']

    if test_name == 'test1':
        dataset = arff.load(open('weather.nominal.arff', 'r', encoding="utf-8"))
        attrs = dataset['attributes']
        attrs = [x[0]  for x in attrs]
        data = pd.DataFrame(dataset['data'], columns = attrs)
        X = data.drop(["play"], axis = 1)
        y = data['play']

    clf = naive_bayes.NaiveBayesClassif()
    scores  = cross_val_scorer(clf, X, y, n = num_of_splits, random_state = r_s)
    return scores

if __name__ == "__main__":
    r_s = 4224
    num_of_splits = 10
    for i in range(1, 3):
        test_name = 'test' + str(i)
        scores = run_test(test_name = test_name, r_s = r_s, num_of_splits = num_of_splits)
        print("Mean accuracy for " + test_name + 
            " for random state " + str(r_s) + 
            " and number of splits " + str(num_of_splits) + 
            ": \n" + str(np.mean(scores))) 