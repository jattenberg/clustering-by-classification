import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin

class ClusterByClassifier(BaseEstimator, ClusterMixin, TransformerMixin):
    
    def __init__(self, clf, n_clusters=8, max_iters=30000, thresh=0.00001):
        self.n_clusters = n_clusters
        self.clf = clf
        self.max_iters = max_iters
        self.thresh = thresh
    
    def fit_predict(self, X, y=None):
        def train_and_predict(y):
            self.clf.fit(X, y)
            return np.array([np.random.choice(self.n_clusters, p=x) for x in self.clf.predict_proba(X)])
        
        def train_rec(y, ct=0):
            y_hat = train_and_predict(y)
            if ct >= self.max_iters or np.linalg.norm(y-y_hat) <= self.thresh:
                return y_hat
            else:
                return train_rec(y_hat, ct+1)
        
        return train_rec(np.random.randint(0, self.n_clusters, size=len(X)))
    
    def fit(self, X, y=None):
        self.fit_predict(X, y)
        
        return self
    
    def transform(self, X, y=None):
        """
        returns the distance to clusters, 
        where i define distance to be 1-proba
        """
        return 1.0 - self.clf.predict_proba(X)
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
        
    def score(self, X, y=None):
        return self.clf.predict_proba(X)
