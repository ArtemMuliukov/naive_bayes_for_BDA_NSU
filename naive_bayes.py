import numpy as np
import pandas as pd
from collections import defaultdict, Counter


class NaiveBayesClassif:

    def __init__(self):
        self.x_forY_probs = defaultdict(lambda: defaultdict(lambda : defaultdict(lambda : 1e-10)))
        self.x_probs = defaultdict(lambda : defaultdict(lambda : 1e-10))
        self.y_probs = defaultdict(lambda : 1e-10)
        

    def fit(self,X, y):
        self.__init__()
        self.counter_y = Counter(y)
        s = len(y)
        for y_name in self.counter_y.keys():
            self.y_probs[y_name] =self.counter_y[y_name] / s
        for i in range(X.shape[1]):
            counter_x = Counter(X[X.columns[i]])
            s = sum(counter_x.values())
            for x_name in counter_x.keys():
                self.x_probs[i][x_name] =counter_x[x_name] / s
        for y_name in self.counter_y.keys():
            mask = (y == y_name)
            for i in range(X.shape[1]):
                counter_x = Counter(X[X.columns[i]][mask])
                s = sum(counter_x.values())
                for x_name in counter_x.keys():
                    self.x_forY_probs[y_name][i][x_name] = counter_x[x_name] / s
         

    def _predict_prob_for_x(self,x,y):
        prob = self.y_probs[y]
        for i,x_cur in enumerate(x):
            prob = prob * self.x_forY_probs[y][i][x_cur]
        return prob
    

    def predict_proba(self, X):
        X_ans = np.zeros((X.shape[0],len(self.y_probs.keys())))
        y_arr = []     
        for y_num, y_name in enumerate(self.y_probs.keys()):
            for i in range(X.shape[0]):
                x = X.values[i]
                pr = self._predict_prob_for_x(x,y_name)
                X_ans[i][y_num] = pr
            y_arr.append(y_name)
        return X_ans/np.array([np.sum(X_ans,axis=1)]).T,y_arr
    

    def predict(self, X):
        ans = []
        probs, y_s = self.predict_proba(X) 
        return ([y_s[i] for i in np.argmax(probs, axis = 1)])
    

    def get_probs(self):
        return self.x_forY_probs
    