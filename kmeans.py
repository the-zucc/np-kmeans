import numpy as np

CONST_POINT_INIT='points'
CONST_UNIF_INIT='unif'

class KMeans:
    def __init__(self, K=3, init=CONST_POINT_INIT):
        self.K = K
        self.labels = None
        self.old_labels = None
        self.init = init
    
    def init_means(self, X, K):
        labels = self.assign_random_labels(X, K)
        mu = self.compute_mu(X,K,labels)
        print("Initial mu shape:")
        print(mu.shape)
        return mu
    
    def init_means_points(self, X, K):
        return np.array(X[np.floor(np.random.random(size=K)*X.shape[0]).astype('int')])

    def compute_mu(self, X, K, labels):
        return np.array([X[labels == i].mean(axis=0) for i in range(self.K)])
    
    def predict(self, X):
        return (
            ((X[:,:,np.newaxis] - self.mu.T)**2).sum(axis=1) ** (1/2)
        ).argmin(axis=1)

    def assign_random_labels(self, X, K):
        return np.floor(np.random.random(size=X.shape[0])*K)
    
    def iteration(self, X, K):
        if self.labels is not None:
            self.old_labels = self.labels.copy()
        self.labels = self.predict(X)

        self.old_mu = self.mu.copy()
        self.mu = self.compute_mu(X, K, self.labels)
        print(self.mu.shape)
        return not (self.old_labels is None or (self.labels != self.old_labels).any())

    def fit(self, X):
        self.X = X
        if self.init == CONST_POINT_INIT:
            self.mu = self.init_means_points(self.X, self.K)
        elif self.init == CONST_UNIF_INIT:
            self.mu = self.init_means(self.X, self.K)
        else
            self.mu = self.init_means_points(self.X, self.K)
            case _:
                print('Warning: incorrect value specified for \'init\': {}', self.init)
        while not self.iteration(self.X, self.K):
            print(self.mu)