""" An example of simple gaussian process regression """
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.models import GP
from src.kernels import rbf

sns.set()
SAVE_FOLDER = "./results"
DATA_FOLDER = "./data"


def main():

    # Initialise the model
    model = GP(covariance=rbf, sigma=1e-10)

    N = 10
    sigma = 0.1

    def gen_data(N, sigma):
        x = np.linspace(-10, 10, N)
        y = np.sin(x) + x + np.random.randn(N)*sigma
        x = np.reshape(x, (x.shape[0], 1))
        y = np.reshape(y, (y.shape[0], 1))
        return x, y

    x, y = gen_data(N, sigma)

    # Fit the model to the data and plot the posterior mean
    model.fit(x, y)
    plt.figure(figsize=(5, 5))
    X = np.linspace(-10, 10,)
    post_mean = model.poster_mean(X)
    plt.plot(x, y, 'ro')
    plt.plot(X, post_mean)
    plt.title("Posterior Mean")
    plt.savefig(SAVE_FOLDER + "/posteriormean.png")



if __name__ == "__main__":
    main()
