""" An example of simple gaussian process regression """
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.models import GP
from src.kernels import rbf, periodic, polynomial

sns.set()
SAVE_FOLDER = "./results"
DATA_FOLDER = "./data"


def main():

    # Initialise the model
    model = GP(covariance=polynomial, sigma=0.1)

    # Generate some fake training data
    N = 100
    sigma = 0.1
    def gen_data(N, sigma):
        x = np.linspace(-5, 5, N)
        y = np.sin(x) + np.random.randn(N)*sigma
        x = np.reshape(x, (x.shape[0], 1))
        y = np.reshape(y, (y.shape[0], 1))
        return x, y

    x, y = gen_data(N, sigma)

    # Fit the model to the data and plot the posterior mean
    model.fit(x, y)
    plt.figure(figsize=(20, 5))
    X = np.linspace(-10, 10, 50)
    ax = plt.subplot(1, 4, 1)
    ax.set_title("Samples from the prior")
    for i in range(2):
        y_prior = model.sample_prior(X)
        plt.plot(X, y_prior)
    ax = plt.subplot(1, 4, 2)
    plt.plot(x, y, 'ro')
    ax.set_title("training data")
    post_mean = model.poster_mean(X)
    var = np.reshape(np.diag(model.posterior_cov(X)), post_mean.shape)
    upper_bound = post_mean + var
    lower_bound = post_mean - var
    ax = plt.subplot(1, 4, 3)
    plt.plot(x, y, 'ro')
    plt.plot(X[10:-10], post_mean[10:-10])
#    plt.fill_between(X, lower_bound[:, 0], upper_bound[:, 0], color='red', alpha=0.1)
    ax.set_title("Posterior Mean and Uncertainty")
    ax = plt.subplot(1, 4, 4)
    for i in range(10):
        samp = model.sample_posterior(X)
        plt.plot(X, samp)
    ax.set_title("Posterior Samples")
    plt.suptitle("Gaussian Process Regression with an exponentiated quadratic kernel")
    plt.savefig(SAVE_FOLDER + "/gpregression.png")
    plt.show()



if __name__ == "__main__":
    main()
