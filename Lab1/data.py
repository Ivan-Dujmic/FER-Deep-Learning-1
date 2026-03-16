import numpy as np
import matplotlib.pyplot as plt

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

        Returns: data points
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

def sample_gmm_2d(K, C, N, parts=1):
    """
    Parameters:
        K - # of distributions
        C - # of classes
        N - # of samples per distribution
        parts - split into multiple sets (useful for train + test sets)

    Returns:
        X - data
        Y_ - classes

        or

        data_parts - parts # of pairs of X and Y_
    """

    Gs = []
    Ys = []
    for _ in range(K):
        Gs.append(Random2DGaussian())
        Ys.append(np.random.randint(C))

    X_samples = [G.get_sample(N * parts) for G in Gs]
    Y_samples = [[Y] * (N * parts) for Y in Ys]

    if parts == 1:
        X = np.vstack(X_samples)
        Y_ = np.hstack(Y_samples)
        return X, Y_
    
    else:
        data_parts = []
        for p in range(parts):
            X_part = np.vstack([X[p * N : (p + 1) * N] for X in X_samples])
            Y_part = np.hstack([Y[p * N : (p + 1) * N] for Y in Y_samples])
            data_parts.append((X_part, Y_part))
        return data_parts

def eval_perf_multi(Y, Y_):
    """
    Parameters:
        Y - predicted classes
        Y_ - true classes

    Returns:
        accuracy
        confusion matrix
        precisions
        recalls
    """

    n_classes = max(Y.max(), Y_.max()) + 1
    conf = np.zeros((n_classes, n_classes), dtype=int)

    for t, p in zip(Y_, Y):
        conf[t, p] += 1

    accuracy = np.trace(conf) / np.sum(conf)

    precision = np.zeros(n_classes)
    for i in range(n_classes):
        sum = np.sum(conf[:, i])
        if sum > 0:
            precision[i] = conf[i, i] / sum

    recall = np.zeros(n_classes)
    for i in range(n_classes):
        sum = np.sum(conf[i, :])
        if sum > 0:
            recall[i] = conf[i, i] / sum

    return accuracy, conf, precision, recall

def average_precision(ranked_labels):
    """
    Returns: Average precision from ranked labels
    """
    n = len(ranked_labels)
    pos = sum(ranked_labels)
    neg = n - pos

    tp = pos
    tn = 0
    fn = 0
    fp = neg

    sumprec=0
    
    for x in ranked_labels:
        precision = tp / (tp + fp) 

        if x:
            sumprec += precision

        tp -= x
        fn += x
        fp -= not x
        tn += not x

    return sumprec / pos

if __name__=="__main__":
    def myDummyDecision(X):
        scores = X[:,0] + X[:,1] - 5
        return scores

    np.random.seed(100)

    X, Y_ = sample_gmm_2d(4, 2, 30)

    Y = myDummyDecision(X) > 0.5  

    rect = (np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(myDummyDecision, rect, offset=0)

    graph_data(X, Y_, Y)

    plt.show()