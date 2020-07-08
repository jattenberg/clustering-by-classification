import logging
import numpy as np
from enum import Enum
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.cluster import KMeans

class initializer(Enum):
    """
    how do we initialize the clustering for
    subsequent cluster by classification
    """
    random="random"
    kmeans="kmeans"
    

class ClusterByClassifier(BaseEstimator, ClusterMixin, TransformerMixin):
    
    def __init__(self,
                 clf,
                 n_clusters=8,
                 max_iters=500,
                 thresh=0.00001,
                 initializer="random",
                 soft_clustering=True):
        self.n_clusters = n_clusters
        self.clf = clf
        self.max_iters = max_iters
        self.thresh = thresh
        self.initializer = initializer
        self.soft_clustering = soft_clustering

    def _initialize(self, X):
        def _random(X):
            return np.random.randint(0, self.n_clusters, size=len(X))

        def _kmeans(X):
            return KMeans(n_clusters=self.n_clusters).fit_predict(X)

        logging.info("initializing with %s" % self.initializer)
        return {
            "random": _random,
            "kmeans": _kmeans
            }[self.initializer](X)
        
    def fit_predict(self, X, y=None):
        def train_and_predict(y):
            self.clf.fit(X, y)
            return np.array([np.random.choice(self.n_clusters, p=x) for x in self.clf.predict_proba(X)])
        
        def train_rec(y, ct=0):
            logging.info("on iteration %d" % ct)

            y_hat = train_and_predict(y)
            # absolute magnitude
            if ct >= self.max_iters or np.linalg.norm(y-y_hat) <= self.thresh:
                return y_hat
            else:
                return train_rec(y_hat, ct+1)
            
        return train_rec(self._initialize(X))
    
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
