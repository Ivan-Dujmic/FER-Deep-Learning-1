import numpy as np
import matplotlib.pyplot as plt
import matplotlib

class Random2DGaussian:
    def __init__(self, x_min=0, x_max=10, y_min=0, y_max=10, cov_scale=0.2):
        """
        Parameters:
            x_min, x_max, y_min, y_max - range where mean can be placed
            cov_scale - spread as a fraction of the total range; [0, 1]
        """

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
        """
        Parameters:
            n - # of samples

        Returns:
            data points
        """

        return np.random.multivariate_normal(self.mean, self.covariance_matrix, n)

def graph_data(X, Y_, Y, special=[]):
    """
    Point colors represent true classes.
    Circle points represent correct guesses.
    Square points represent incorrect guesses.

    Parameters:
        X - data
        Y_ - true classes
        Y - predicted classes
        special - emphasize points
    """
    colors_list = np.array([
        "#ff0000",
        "#00ff00",
        "#0000ff",
        "#ffff00",
        "#00ffff",
        "#ff00ff",
        "#ffffff",
        "#000000"
    ])

    num_classes = np.max(Y_) + 1
    colors = colors_list[Y_ % len(colors_list)]

    sizes = np.repeat(30, len(Y_))
    sizes[special] = 60

    correct = (Y == Y_)
    plt.scatter(X[correct, 0], X[correct, 1], c=colors[correct], s=sizes[correct], marker='o', edgecolors='black')

    incorrect = (Y != Y_)
    plt.scatter(X[incorrect, 0], X[incorrect, 1], c=colors[incorrect], s=sizes[incorrect], marker='s', edgecolors='black')

def graph_surface(fun, rect, offset=0.5, width=256, height=256):
    """
    Parameters:
        fun - decision function
        rect - domain ([x_min, y_min], [x_max, y_max])
        offset - color palette offset
        width, height - resolution
    """

    lin_x = np.linspace(rect[0][0], rect[1][0], width)
    lin_y = np.linspace(rect[0][1], rect[1][1], height)
    xx, yy = np.meshgrid(lin_x, lin_y)
    grid = np.column_stack((xx.flatten(), yy.flatten()))

    values = fun(grid).reshape(height, width)
    maxval = max(np.max(values) - offset, offset - np.min(values))

    plt.pcolormesh(xx, yy, values, vmin=offset - maxval, vmax=offset + maxval)
    
    plt.contour(xx, yy, values, colors='black', levels=[offset])

def sample_gmm_2d(K, C, N):
    """
    Parameters:
        K - # of distributions
        C - # of classes
        N - # of samples per distribution

    Returns:
        X - data
        Y_ - classes
    """

    Gs = []
    Ys = []
    for _ in range(K):
        Gs.append(Random2DGaussian())
        Ys.append(np.random.randint(C))

    X = np.vstack([G.get_sample(N) for G in Gs])
    Y_ = np.hstack([[Y] * N for Y in Ys])

    return X, Y_

if __name__=="__main__":
    # TEST ________________________
    def dummy(X):
        np.random.seed(100)
        return np.random.randint(0, 3, 5 * 25)

    def myDummyDecision(X):
        scores = X[:,0] + X[:,1] - 5
        return scores

    np.random.seed(100)

    # get data
    X,Y_ = sample_gmm_2d(4, 2, 30)
    # X,Y_ = sample_gauss_2d(2, 100)

    # get the class predictions
    Y = myDummyDecision(X)>0.5  

    # graph the decision surface
    rect=(np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(myDummyDecision, rect, 0.5)

    # graph the data points
    graph_data(X, Y_, Y, special=[])

    plt.show()