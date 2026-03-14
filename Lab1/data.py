import numpy as np
import matplotlib.pyplot as plt

class Random2DGaussian:
    def __init__(self, x_min = 0, x_max = 10, y_min = 0, y_max = 10, cov_scale = 0.2):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.cov_scale = cov_scale

        self.mean = np.array([np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)])
        eigenvalues = np.random.uniform([x_min, y_min], [x_max, y_max])
        eigenvalues *= cov_scale
        eigenvalues **= 2
        angle = np.random.uniform(0, 2*np.pi)
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        self.covariance_matrix = np.dot(np.dot(np.transpose(rotation_matrix), np.diag(eigenvalues)), rotation_matrix)

    def get_sample(self, n):
        return np.random.multivariate_normal(self.mean, self.covariance_matrix, n)

# K - # of distributions
# C - # of classes
# N - # of samples per distribution 
def sample_gmm_2d(K, C, N):
    Gs = []
    Ys = []
    for _ in range(K):
        Gs.append(Random2DGaussian())
        Ys.append(np.random.randint(C))

    X = np.vstack([G.get_sample(N) for G in Gs])
    Y_ = np.hstack([[Y] * N for Y in Ys])

    return X, Y_

np.random.seed(100)
G=Random2DGaussian()
X=G.get_sample(100)
plt.scatter(X[:,0], X[:,1])
plt.show()