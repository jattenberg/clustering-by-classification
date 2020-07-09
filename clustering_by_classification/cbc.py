import logging
import numpy as np
from enum import Enum
from annoy import AnnoyIndex
from scipy.special import softmax
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
                 initializer="modelmkpp",
                 soft_clustering=True,
                 neighbor_selection='balanced'):
        self.n_clusters = n_clusters
        self.clf = clf
        self.max_iters = max_iters
        self.thresh = thresh
        self.initializer = initializer
        self.soft_clustering = soft_clustering
        self.neighbor_selection = neighbor_selection

    def _initialize(self, X):
        def _get_num_neighbors():
            init = self.neighbor_selection
            if isinstance(init, int) and init > 0:
                return init
            elif isinstance(init, str) and init is "balanced":
                return int(X.shape[0]/self.n_clusters)
            elif isinstance(init, str) and init is "halfbalanced":
                return int(X.shape[0]/(2*self.n_clusters))
            else:
                raise ValueError("unsupported neighbor selection %s" % init)

        def _random(X):
            return np.random.randint(0, self.n_clusters, size=len(X))

        def _random_nn(X):    
            idx = AnnoyIndex(X.shape[1], 'euclidean')
            for i in range(X.shape[0]):
                idx.add_item(i, X[i])

            logging.info("building an index with %d items" % X.shape[0])
            idx.build(50)

            logging.info("finding %d neighbor groups" % self.n_clusters)
            seen = {}
            label = 0

            guess = np.random.randint(X.shape[0])
            centers = {guess:0}
            
            while label < self.n_clusters:
                neighbors = idx.get_nns_by_item(guess, _get_num_neighbors())
                for point in neighbors:
                    seen[point] = label
                seen[guess] = label

                # find a distant point
                dists = np.array([
                    [idx.get_distance(i, j) for i in centers]
                    for j in range(X.shape[0])
                    ])

                avg_dists = np.average(dists, axis=1)
                dist_prob = softmax(avg_dists)
                
                guess = np.random.choice(X.shape[0], p=dist_prob)
                
                while guess in seen:
                    guess = np.random.choice(X.shape[0], p=dist_prob)
                centers[guess] = label
                     
                label = label + 1

            y = np.zeros(X.shape[0])

            for k, v in seen.items():
                y[k] = v
            return y
                
        def _kmeans(X):
            return KMeans(n_clusters=self.n_clusters).fit_predict(X)

        def _modelmkpp(X):
            idx = AnnoyIndex(X.shape[1], 'euclidean')
            for i in range(X.shape[0]):
                idx.add_item(i, X[i])

            logging.info("building an index with %d items" % X.shape[0])
            idx.build(50)

            label = 1
            first = np.random.randint(X.shape[0])
            
            seen = {first:label}
            
            while len(seen) < self.n_clusters - 1:
                label=label+1
                ys = np.zeros(X.shape[0])

                # 1. find rough groups to train classifiers
                for guess,v in seen.items():
                    to_get = int(X.shape[0]/(len(seen) + 1))
                    neighbors = idx.get_nns_by_item(guess, to_get)
                    for n in neighbors:
                        ys[n] = v
                    ys[guess] = v

                self.clf.fit(X, ys)

                # 2. compute probas for examples:
                probas = self.clf.predict_proba(X)
                # 3. find the max proba for each example
                # todo: something other than max?
                dists = np.max(1.0 - probas, axis=1)
                # 4. sample based on the distribution of dists
                nxt_pnt = np.random.choice(X.shape[0], p=dists/np.sum(dists))

                while nxt_pnt in seen:
                    nxt_pnt = np.random.choice(X.shape[0], p=dists/np.sum(dists))

                seen[nxt_pnt] = label

            ys = np.zeros(X.shape[0])

            for k,v in seen.items():
                to_get = int(X.shape[0]/(len(seen) + 1))
                neighbors = idx.get_nns_by_item(k, to_get)
                for n in neighbors:
                    ys[n] = v
                ys[k] = v

            self.clf.fit(X, ys)
    
            # todo: soft-clustering version
            return self.clf.predict(X)

        logging.info("initializing with %s" % self.initializer)
        return {
            "random": _random,
            "random_nn": _random_nn,
            "kmeans": _kmeans,
            "modelmkpp": _modelmkpp,
            }[self.initializer](X)
        
    def fit_predict(self, X, y=None):
        def train_and_predict(y):
            self.clf.fit(X, y)

            probas = self.clf.predict_proba(X)

            n_clusters = min(self.n_clusters, probas.shape[1])
            if n_clusters != self.n_clusters:
                logging.warn("reduction in number of clusers,from %d to %d"\
                             % (self.n_clusters, n_clusters))
                self.n_clusters = n_clusters
                                     
            if self.soft_clustering:
                return np.array([np.random.choice(self.n_clusters, p=x) for x in probas])
            else:
                return self.clf.predict(X)
        
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
