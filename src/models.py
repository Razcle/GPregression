""" Gaussian Process Regression Models """
import numpy as np

# TODO vectorise gram contsruction


class GP:
    """ Base class for a GP regression model """

    def __init__(self, covariance, mean=None, sigma=0.1):
        self.prior_mean = mean
        self.prior_cov = covariance
        self.sigma = sigma

    def fit(self, X, y):
        """ Calculates the posterior mean and covariance given some function
        evaluations y """
        self.KXX = self._build_gram_matrix(X, self.prior_cov)
        self.KXx = self._build_KXx(X, self.prior_cov)
        self.poster_mean = self._get_posterior_mean(self.KXx, self.KXX, y)
        self.posterior_cov = self._get_posterior_cov(self.KXx, self.KXX, y)

    def predict(self, X):
        if not self.poster_mean:
            print("You must train the model before making predictions!")
            return
        return self.posterior_mean(X)

    def sample_prior(self, X):
        """ Draw a sample of the GP prior function evaluated at X"""
        gram = self._build_gram_matrix(X, self.prior_cov)
        return self._sample(X, gram)

    def sample_posterior(self, X):
        """ Draw a sample from the GP posterior"""
        mean = self.poster_mean(X)
        gram = self.posterior_cov(X)
        return self._sample(X, gram, mean)

    def _build_gram_matrix(self, X, kernel):
        N = X.shape[0]
        gram = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                gram[i, j] = kernel(X[i], X[j])
        gram = gram + self.sigma*np.eye(X.shape[0])
        return gram

    def _build_KXx(self, X, kernel):
        def KX(x):
            N = X.shape[0]
            D = x.shape[0]
            k = np.zeros((N, D))
            for i in range(N):
                for j in range(D):
                    k[i, j] = kernel(X[i], x[j])
            return k
        return KX

    def _sample(self, X, gram, mean=0.0):
        chol = np.linalg.cholesky(gram)
        Z = np.random.randn(X.shape[0], 1)
        samp = mean + np.dot(chol, Z)
        return samp

    def _get_posterior_mean(self, KXx, KXX, y):
        def post_mean(x):
            k = KXx(x)
            v = np.linalg.lstsq(KXX, y)[0]
            return np.dot(k.T, v)
        return post_mean

    def _get_posterior_cov(self, KXx, KXX, y):
        def post_cov(self, x):
            kxx = self.prior_cov(x, x)
            k = KXx(x)
            v = np.linalg.lstsq(KXX, k)[0]
            return kxx - np.dot(k.T, v)
        return post_cov
