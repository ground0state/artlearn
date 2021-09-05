import sys

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.base import BaseEstimator, ClassifierMixin, ClusterMixin
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_array, check_is_fitted

EPS = 1e-10
FLOAT_MAX = sys.float_info.max


def normalize(v):
    v_norm = np.linalg.norm(v)
    if (v_norm == 0) and (EPS <= 0):
        v = np.zeros_like(v)
    else:
        v = v / (v_norm + EPS)
    return v


class ART1(BaseEstimator, ClusterMixin):
    """ART1.

    Parameters
    ----------
    max_iter : int
        Maximum number of iterations of the ART2 algorithm to run.
    max_class : int
        Maximum number of the class to classify.
    rho : float, default=0.75
        Threshold for degree of match.
    L : float, default=0.75
        Larger values of L bias the selection of inactive nodes over active ones.
    alpha : float, default=1e-5
        Choice parameter.
    """

    def __init__(
            self,
            max_iter=10,
            max_class=100,
            rho=0.75,
            L=1.5,
            alpha=1e-5):
        super().__init__()
        self.max_iter = max_iter
        self.max_class = max_class
        self.rho = rho
        self.L = L
        self.alpha = alpha

    def _check_params(self, X):
        # max_iter
        if self.max_iter <= 0:
            raise ValueError(
                f"max_iter must be > 0, got {self.max_iter} instead.")
        # max_class
        if self.max_class <= 0:
            raise ValueError(
                f"max_class must be > 0, got {self.max_class} instead.")
        # rho
        if not (0.0 < self.rho < 1.0):
            raise ValueError(
                f"rho must be in range (0, 1), got {self.rho} instead.")
        # L
        if self.L <= 0:
            raise ValueError(
                f"L must be > 0, got {self.L} instead.")
        # alpha
        if self.alpha <= 0:
            raise ValueError(
                f"alpha must be > 0, got {self.alpha} instead.")

    def _initialize(self, X):
        self.history_ = {"gap": [], "similarity": []}
        self._n_features = X.shape[1]
        self._b = np.ones((self.max_class, self._n_features)) * \
            self.L / (self.L - 1 + self._n_features) * 0.5
        self._t = np.ones((self.max_class, self._n_features))
        self._active_list = [False] * self.max_class

    def fit(self, X):
        """Compute ART1 clustering.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.

        Returns
        -------
        self
            Fitted estimator.
        """
        X = check_array(X)
        self._check_params(X)
        self._initialize(X)
        for _ in range(self.max_iter):
            y = self._resonance_iter(X)
        self.labels_ = y
        return self

    def partial_fit_predict(self, X):
        check_is_fitted(self)
        self.labels_ = self._resonance_iter(X)
        return self.labels_

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
            Returns -1 label if the match class does not exist.
        """
        check_is_fitted(self)
        X = check_array(X)
        return self._resonance_iter(X, should_update=False)

    def _resonance_iter(self, X, should_update=True):
        y = -1 * np.ones(X.shape[0], dtype=np.int32)
        for idx, s in enumerate(X):
            # F1
            T = np.sum(s * self._b, axis=1) / \
                (self.alpha + np.sum(self._b, axis=1))

            # Search match
            for _ in range(self.max_class):
                # F2 : Code selection
                J = np.argmax(T)
                # similarity = How much s is contained in t
                similarity = np.sum(s * self._t[J]) / (self.alpha + np.sum(s))

                # Match
                if (not self._active_list[J]) or (similarity >= self.rho):
                    if should_update:
                        y[idx] = J

                        # Updata parameters
                        self._b[J] = self.L * s * self._t[J] / \
                            (self.L - 1 + np.sum(s * self._t[J]))
                        self._t[J] = s * self._t[J]

                        gap = np.mean(np.abs(self._b - self._t))
                        # logging
                        self.history_["gap"].append(gap)
                        self.history_["similarity"].append(similarity)

                        self._active_list[J] = True
                    else:
                        if self._active_list[J]:
                            y[idx] = J
                    break

                # Do not match
                else:
                    T[J] = 0
                    continue
        return y


class ART2(BaseEstimator, ClusterMixin):
    """ART2.

    Parameters
    ----------
    max_iter : int, default=10
        Maximum number of iterations of the ART2 algorithm to run.
    max_class : int, default=100
        Maximum number of the class to classify.
    rho : float, default=0.95
        Threshold for degree of match.
    a : float, default=0.1
        Degree of retaining memory of previous data.
    b : float, default=0.1
        Degree of retaining memory of the last classified class.
    c : float, default=0.1
        Control the range of degree of match.
    d : float, default=0.9
        Learning rata..
    theta : float, default=0.0
        Noise reduction threshold for normalized data.
    alpha : float, default=None
        Initialization factor of LTM Trace.
    """

    def __init__(
        self,
        max_iter=10,
        max_class=100,
        rho=0.95,
        a=0.1,
        b=0.1,
        c=0.1,
        d=0.9,
        theta=0.0,
            alpha=None):
        super().__init__()
        self.max_iter = max_iter
        self.max_class = max_class
        self.rho = rho
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.theta = theta
        self.alpha = alpha
        self._big_m = 1

    def _check_params(self, X):
        # max_iter
        if self.max_iter <= 0:
            raise ValueError(
                f"max_iter must be > 0, got {self.max_iter} instead.")
        # max_class
        if self.max_class <= 0:
            raise ValueError(
                f"max_class must be > 0, got {self.max_class} instead.")
        # rho
        if not (0.0 < self.rho < 1.0):
            raise ValueError(
                f"rho must be in range (0, 1), got {self.rho} instead.")
        # a
        if self.a < 0:
            raise ValueError(
                f"a must be >= 0, got {self.a} instead.")
        # b
        if self.b < 0:
            raise ValueError(
                f"b must be >= 0, got {self.b} instead.")
        # c
        if self.c <= 0:
            raise ValueError(
                f"c must be > 0, got {self.c} instead.")
        # d
        if not (0.0 < self.d < 1.0):
            raise ValueError(
                f"d must be in range (0, 1), got {self.d} instead.")
        # theta
        if not (0.0 <= self.theta < 1.0):
            raise ValueError(
                f"theta must be in range [0, 1), got {self.theta} instead.")
        # alpha
        alpha = 1 / ((1 - self.d) * X.shape[1]**0.5)
        if self.alpha is None:
            self.alpha = alpha
        elif not (0.0 < self.theta < alpha):
            raise ValueError(
                f"theta must be in range (0, 1/((1-d)*sqrt(M)) ), got {self.alpha} instead.")
        # constraint check
        sigma = self.c * self.d / (1 - self.d)
        if sigma > 1:
            raise ValueError(
                f"cd/(1-d) must be <= 1, got {sigma} instead.")

    def _initialize(self, X):
        self.history_ = {"gap": [], "r_norm": []}
        self._n_features = X.shape[1]
        self._u = np.zeros(self._n_features)
        self._p = np.zeros(self._n_features)
        self._z = self.alpha * np.ones((self._n_features, self.max_class))
        self._active_list = [False] * self.max_class

    def fit(self, X):
        """Compute ART2 clustering.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.

        Returns
        -------
        self
            Fitted estimator.
        """
        # validation
        X = check_array(X)
        self._check_params(X)
        self._initialize(X)

        # learning
        for _ in range(self.max_iter):
            y = self._resonance_iter(X)
        self.labels_ = y
        return self

    def partial_fit_predict(self, X):
        check_is_fitted(self)
        self.labels_ = self._resonance_iter(X)
        return self.labels_

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
            Returns -1 label if the match class does not exist.
        """
        check_is_fitted(self)
        X = check_array(X)
        return self._resonance_iter(X, should_update=False)

    def _resonance_iter(self, X, should_update=True):
        y = -1 * np.ones(X.shape[0], dtype=np.int32)
        for idx, s in enumerate(X):
            # F1 : Update the value of each node
            w = s + self.a * self._u
            x = normalize(w)
            q = normalize(self._p)
            v = self._activate(x) + self.b * self._activate(q)
            u = normalize(v)
            T = np.dot(u, self._z)

            # Search match
            for _ in range(self.max_class):
                # F2 : Code selection
                J = np.argmax(T)

                # Caluculate degree of match
                p = u + self.d * self._z[:, J]
                r = (u + self.c * p) / \
                    (np.linalg.norm(u) + np.linalg.norm(self.c * p))
                r_norm = np.linalg.norm(r)

                # Match
                if (not self._active_list[J]) or (r_norm >= self.rho):
                    self._u = u
                    self._p = p
                    if should_update:
                        y[idx] = J
                        # Updata parameters
                        z_previous = self._z.copy()
                        self._z[:, J] = (1 - self.d) * \
                            self._z[:, J] + self.d * self._p
                        gap = np.mean(np.abs(z_previous - self._z))
                        # logging
                        self.history_["gap"].append(gap)
                        self.history_["r_norm"].append(r_norm)
                        self._active_list[J] = True

                    else:
                        if self._active_list[J]:
                            y[idx] = J
                    break

                # Do not match
                else:
                    T[J] = - self._big_m
                    continue
        return y

    def _activate(self, x):
        x[np.abs(x) < self.theta] = 0
        return x


class ART2A(BaseEstimator, ClusterMixin):
    """ART2-A.

    Parameters
    ----------
    max_iter : int
        Maximum number of iterations of the ART2 algorithm to run.
    max_class : int
        Maximum number of the class to classify.
    rho_star : float, default=0.95
        Threshold for degree of match.
    eta : float, default=0.1
        Learning rata.
    theta : float, default=0.0
        Noise reduction threshold for normalized data.
    alpha : float, default=None
        Initialization factor of LTM Trace.
    """

    def __init__(
            self,
            max_iter=10,
            max_class=100,
            rho_star=0.95,
            eta=0.1,
            theta=0.01,
            alpha=None):
        super().__init__()
        self.max_iter = max_iter
        self.max_class = max_class
        self.theta = theta
        self.alpha = alpha
        self.eta = eta
        self.rho_star = rho_star
        self._big_m = 1

    def _check_params(self, X):
        # max_iter
        if self.max_iter <= 0:
            raise ValueError(
                f"max_iter must be > 0, got {self.max_iter} instead.")
        # max_class
        if self.max_class <= 0:
            raise ValueError(
                f"max_class must be > 0, got {self.max_class} instead.")
        # rho
        if not (0.0 < self.rho_star < 1.0):
            raise ValueError(
                f"rho must be in range (0, 1), got {self.rho} instead.")
        # eta
        if not (0.0 < self.eta < 1.0):
            raise ValueError(
                f"eta must be in range (0, 1), got {self.eta} instead.")
        # theta
        if not (0.0 <= self.theta < 1.0):
            raise ValueError(
                f"theta must be in range [0, 1), got {self.theta} instead.")
        # alpha
        alpha = 1 / (X.shape[1]**0.5)
        if self.alpha is None:
            self.alpha = alpha
        elif not (0.0 < self.theta <= alpha):
            raise ValueError(
                f"theta must be in range (0, 1/sqrt(M)], got {self.alpha} instead.")

    def _initialize(self, X):
        self.history_ = {"gap": [], "similarity": []}
        self._n_features = X.shape[1]
        self._z = self.alpha * np.ones((self._n_features, self.max_class))
        self._active_list = [False] * self.max_class

    def fit(self, X):
        """Compute ART2 clustering.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.

        Returns
        -------
        self
            Fitted estimator.
        """
        X = check_array(X)
        self._check_params(X)
        self._initialize(X)
        for _ in range(self.max_iter):
            y = self._resonance_iter(X)
        self.labels_ = y
        return self

    def partial_fit_predict(self, X):
        check_is_fitted(self)
        self.labels_ = self._resonance_iter(X)
        return self.labels_

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
            Returns -1 label if the match class does not exist.
        """
        check_is_fitted(self)
        X = check_array(X)
        return self._resonance_iter(X, should_update=False)

    def _resonance_iter(self, X, should_update=True):
        y = -1 * np.ones(X.shape[0], dtype=np.int32)
        for idx, s in enumerate(X):
            # F1 : Update the value of each node
            u = normalize(self._activate(normalize(s)))
            T = np.dot(u, self._z)

            # Search match
            for _ in range(self.max_class):
                # F2 : Code selection
                J = np.argmax(T)

                # Match
                if (not self._active_list[J]) or (T[J] >= self.rho_star):
                    if should_update:
                        y[idx] = J

                        # Updata parameters
                        z_previous = self._z.copy()
                        if self._active_list[J]:
                            psi = u.copy()
                            psi[self._z[:, J] <= self.theta] = 0
                            self._z[:, J] = normalize(
                                self.eta * psi + (1 - self.eta) * self._z[:, J])
                        else:
                            self._z[:, J] = u

                        gap = np.mean(np.abs(z_previous - self._z))
                        # logging
                        self.history_["gap"].append(gap)
                        self.history_["similarity"].append(T[J])
                        self._active_list[J] = True
                    else:
                        if self._active_list[J]:
                            y[idx] = J
                    break

                # Do not match
                else:
                    T[J] = - self._big_m
                    continue
        return y

    def _activate(self, x):
        x[np.abs(x) < self.theta] = 0
        return x


class FuzzyART(BaseEstimator, ClusterMixin):
    """FuzzyART.

    Parameters
    ----------
    max_iter : int
        Maximum number of iterations of the ART2 algorithm to run.
    max_class : int
        Maximum number of the class to classify.
    rho : float, default=0.75
        Threshold for degree of match.
    alpha : float, default=1e-5
        Choice parameter.
    beta : float, default=0.1
        Learning rata.
    """

    def __init__(
            self,
            max_iter=10,
            max_class=100,
            rho=0.75,
            alpha=1e-5,
            beta=0.1):
        super().__init__()
        self.max_iter = max_iter
        self.max_class = max_class
        self.rho = rho
        self.alpha = alpha
        self.beta = beta

        self._big_m = 1

    def _check_params(self, X):
        # max_iter
        if self.max_iter <= 0:
            raise ValueError(
                f"max_iter must be > 0, got {self.max_iter} instead.")
        # max_class
        if self.max_class <= 0:
            raise ValueError(
                f"max_class must be > 0, got {self.max_class} instead.")
        # rho
        if not (0.0 < self.rho < 1.0):
            raise ValueError(
                f"rho must be in range (0, 1), got {self.rho} instead.")
        # alpha
        if self.alpha <= 0:
            raise ValueError(
                f"alpha must be > 0, got {self.alpha} instead.")
        # beta
        if not (0.0 < self.beta < 1.0):
            raise ValueError(
                f"beta must be in range (0, 1), got {self.beta} instead.")

    def _initialize(self, X):
        self.history_ = {"gap": [], "similarity": []}
        self._n_features = X.shape[1]
        self._w = np.ones((self.max_class, self._n_features))
        self._active_list = [False] * self.max_class

    def fit(self, X):
        """Compute FuzzyART clustering.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.

        Returns
        -------
        self
            Fitted estimator.
        """
        X = check_array(X)
        self._check_params(X)
        self._initialize(X)
        for _ in range(self.max_iter):
            y = self._resonance_iter(X)
        self.labels_ = y
        return self

    def partial_fit_predict(self, X):
        check_is_fitted(self)
        self.labels_ = self._resonance_iter(X)
        return self.labels_

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
            Returns -1 label if the match class does not exist.
        """
        check_is_fitted(self)
        X = check_array(X)
        return self._resonance_iter(X, should_update=False)

    def _resonance_iter(self, X, should_update=True):
        y = -1 * np.ones(X.shape[0], dtype=np.int32)
        for idx, s in enumerate(X):
            # F1
            T = np.sum(np.minimum(s, self._w), axis=1) / \
                (self.alpha + np.sum(self._w, axis=1))

            # Search match
            for _ in range(self.max_class):
                # F2 : Code selection
                J = np.argmax(T)
                similarity = np.sum(np.minimum(s, self._w[J])) / np.sum(s)

                # Match
                if (not self._active_list[J]) or (similarity >= self.rho):
                    if should_update:
                        y[idx] = J

                        # Updata parameters
                        w_previous = self._w.copy()
                        if self._active_list[J]:
                            self._w[J] = self.beta * \
                                np.minimum(s, self._w[J]) + (1 - self.beta) * self._w[J]
                        else:
                            self._w[J] = np.minimum(s, self._w[J])

                        gap = np.mean(np.abs(w_previous - self._w))
                        # logging
                        self.history_["gap"].append(gap)
                        self.history_["similarity"].append(similarity)

                        self._active_list[J] = True

                    else:
                        if self._active_list[J]:
                            y[idx] = J
                    break

                # Do not match
                else:
                    T[J] = - self._big_m
                    continue
        return y


class SFAM(BaseEstimator, ClassifierMixin):
    """SFAM.

    Parameters
    ----------
    max_iter : int
        Maximum number of iterations of the ART2 algorithm to run.
    max_class : int
        Maximum number of the class to classify.
    rho : float, default=0.75
        Threshold for degree of match.
    alpha : float, default=1e-5
        Choice parameter.
    beta : float, default=0.1
        Learning rata.
    epsilon : float, default=0.0001
        Paramter to increase vigilance rho.
    """

    def __init__(
            self,
            max_iter=10,
            max_class=100,
            rho=0.75,
            alpha=1e-5,
            beta=0.1,
            epsilon=0.0001):
        super().__init__()
        self.max_iter = max_iter
        self.max_class = max_class
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

        self._big_m = 1

    def _check_params(self, X):
        # max_iter
        if self.max_iter <= 0:
            raise ValueError(
                f"max_iter must be > 0, got {self.max_iter} instead.")
        # max_class
        if self.max_class <= 0:
            raise ValueError(
                f"max_class must be > 0, got {self.max_class} instead.")
        # rho
        if not (0.0 < self.rho < 1.0):
            raise ValueError(
                f"rho must be in range (0, 1), got {self.rho} instead.")
        # alpha
        if self.alpha <= 0:
            raise ValueError(
                f"alpha must be > 0, got {self.alpha} instead.")
        # beta
        if not (0.0 < self.beta < 1.0):
            raise ValueError(
                f"beta must be in range (0, 1), got {self.beta} instead.")
        # epsilon
        if self.epsilon <= 0:
            raise ValueError(
                f"epsilon must be > 0, got {self.epsilon} instead.")

    def _initialize(self, X):
        self.history_ = {"gap": [], "similarity": []}
        self._n_features = X.shape[1]
        self._w = np.ones((self.max_class, self._n_features))
        self._active_list = [False] * self.max_class
        self._inner_label2y = dict()

    def fit(self, X, y):
        """Compute SFAM clustering.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
        y : array-like of shape (n_samples, )
            Label vector relative to X.

        Returns
        -------
        self
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        self._check_params(X)
        self._initialize(X)
        for _ in range(self.max_iter):
            self._resonance_iter(X, y)
        return self

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.
        Returns
        -------
        y : ndarray of shape (n_samples,)
            Class labels for samples in X.
            Returns -1 label if the match class does not exist.
        """
        check_is_fitted(self)
        X = check_array(X)
        y = -1 * np.ones(X.shape[0], dtype=np.int32)

        for idx, s in enumerate(X):
            T = np.sum(np.minimum(s, self._w), axis=1) / \
                (self.alpha + np.sum(self._w, axis=1))

            for _ in range(self.max_class):
                J = np.argmax(T)
                similarity = np.sum(np.minimum(s, self._w[J])) / np.sum(s)
                # Do resonance
                if (self._active_list[J]) and (similarity >= self.rho):
                    # Do resonance
                    y[idx] = self._inner_label2y[J]
                # inactive or dont resonance
                else:
                    T[J] = - self._big_m
                    continue

        return y

    def _resonance_iter(self, X, y):
        for idx, s in enumerate(X):
            # F1
            rho = self.rho
            T = np.sum(np.minimum(s, self._w), axis=1) / \
                (self.alpha + np.sum(self._w, axis=1))
            w_previous = self._w.copy()

            # Search match
            for _ in range(self.max_class):
                # Data mismatch
                if rho > 1:
                    break

                # F2 : Code selection
                J = np.argmax(T)
                similarity = np.sum(np.minimum(s, self._w[J])) / np.sum(s)

                # active
                if self._active_list[J]:
                    # Do resonance
                    if similarity >= self.rho:
                        # Class match
                        if self._inner_label2y[J] == y[idx]:
                            self._w[J] = self.beta * \
                                np.minimum(s, self._w[J]) + (1 - self.beta) * self._w[J]
                            break

                        # Class does not match
                        else:
                            T[J] = - self._big_m
                            rho += self.epsilon
                            continue

                    # Do not resonance
                    else:
                        T[J] = - self._big_m
                        continue

                # inactive
                else:
                    self._inner_label2y[J] = y[idx]
                    self._w[J] = np.minimum(s, self._w[J])
                    self._active_list[J] = True
                    break
            else:
                pass
            # break then logging
            gap = np.mean(np.abs(w_previous - self._w))
            self.history_["gap"].append(gap)
            self.history_["similarity"].append(similarity)


class BayesianART(BaseEstimator, ClusterMixin):
    """BayesianART.

    Parameters
    ----------
    max_iter : int
        Maximum number of iterations of the ART2 algorithm to run.
    max_class : int
        Maximum number of the class to classify.
    rho : float, default=0.75
        Threshold for degree of match.
    sigma : float, default=0.1
        Choice parameter.
    max_hyper_volume : float, default=None
        Maximum allowed hyper-volume of classes.
    """

    def __init__(self, max_iter=10, max_class=100, rho=0.75, sigma=0.1, max_hyper_volume=FLOAT_MAX):
        super().__init__()
        self.max_iter = max_iter
        self.max_class = max_class
        self.rho = rho
        self.sigma = sigma
        self.max_hyper_volume = max_hyper_volume

        self._big_m = 1

    def _check_params(self, X):
        # max_iter
        if self.max_iter <= 0:
            raise ValueError(
                f"max_iter must be > 0, got {self.max_iter} instead.")
        # max_class
        if self.max_class <= 0:
            raise ValueError(
                f"max_class must be > 0, got {self.max_class} instead.")
        # rho
        if not (0.0 < self.rho < 1.0):
            raise ValueError(
                f"rho must be in range (0, 1), got {self.rho} instead.")
        # sigma
        if self.sigma <= 0:
            raise ValueError(
                f"sigma must be > 0, got {self.sigma} instead.")
        # max_hyper_volume
        if self.max_hyper_volume <= 0:
            raise ValueError(
                f"max_hyper_volume must be > 0, got {self.max_hyper_volume} instead.")

    def _initialize(self, X):
        self.history_ = {"gap_mu": [], "gap_cov": [], "likelihood": [], "hyper_volume": []}
        self._n_features = X.shape[1]
        self._mu = np.zeros((self.max_class, self._n_features))
        self._cov = np.repeat(
            np.diag([self.sigma**2] * self._n_features)[None, ...], self.max_class, axis=0)
        self._counter = np.array([0] * self.max_class)

        # constraint
        if self.sigma**2 > np.power(self.rho, 1 / self._n_features):
            raise ValueError(
                f"sigma**2 must be < rho**(1/n_feature), sigma**2={self.sigma**2}, rho**(1/n_feature)={np.power(self.rho, 1 / self._n_features)}")

    def fit(self, X):
        """Compute FuzzyART clustering.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.

        Returns
        -------
        self
            Fitted estimator.
        """
        X = check_array(X)
        self._check_params(X)
        self._initialize(X)
        for _ in range(self.max_iter):
            y = self._resonance_iter(X)
        self.labels_ = y
        return self

    def partial_fit_predict(self, X):
        check_is_fitted(self)
        self.labels_ = self._resonance_iter(X)
        return self.labels_

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
            Returns -1 label if the match class does not exist.
        """
        check_is_fitted(self)
        X = check_array(X)
        return self._resonance_iter(X, should_update=False)

    def _gaussian(self, x, j):
        return multivariate_normal.pdf(x, mean=self._mu[j], cov=self._cov[j])

    def _resonance_iter(self, X, should_update=True):
        y = -1 * np.ones(X.shape[0], dtype=np.int32)

        # first trial
        begin = 0
        if self._counter[0] == 0:
            self._mu[0] = X[0].copy()
            self._counter[0] = 1
            begin = 1

        for idx in range(begin, len(X)):
            s = X[idx].copy()
            # F1
            T_denominator = np.sum(
                [
                    self._gaussian(
                        s,
                        j) *
                    self._counter[j] /
                    np.sum(
                        self._counter) for j in np.where(
                        self._counter > 0)[0]])
            T = np.array([self._gaussian(s, j) *
                          self._counter[j] /
                          np.sum(self._counter) /
                          T_denominator for j in np.where(self._counter > 0)[0]])

            # Search match
            for _ in range(len(T)):
                # F2 : Code selection
                J = np.argmax(T)
                likelihood = self._gaussian(s, J)
                hyper_volume = np.power(np.linalg.det(self._cov[J]), 1 / (2 * self._n_features))

                # Match
                if ((likelihood >= self.rho) and (hyper_volume <= self.max_hyper_volume)):
                    y[idx] = J
                    if should_update:
                        mu_previous = self._mu.copy()
                        cov_previous = self._cov.copy()
                        # Updata parameters
                        self._counter[J] += 1
                        self._mu[J] = (1 - 1 / self._counter[J]) * \
                            self._mu[J] + 1 / self._counter[J] * s
                        self._cov[J] = (1 - 1 / self._counter[J]) * self._cov[J] + 1 / \
                            self._counter[J] * np.outer(s - self._mu[J], s - self._mu[J])

                        gap_mu = np.mean(np.abs(mu_previous - self._mu))
                        gap_cov = np.mean(np.abs(cov_previous - self._cov))
                        # logging
                        self.history_["gap_mu"].append(gap_mu)
                        self.history_["gap_cov"].append(gap_cov)
                        self.history_["likelihood"].append(likelihood)
                        self.history_["hyper_volume"].append(hyper_volume)
                    break

                # Do not match
                else:
                    T[J] = - self._big_m
                    continue
            # create new class
            else:
                new_class_idx = len(T)
                if new_class_idx < self.max_class:
                    self._counter[new_class_idx] = 1
                    self._mu[new_class_idx] = s
                else:
                    pass
        return y


class L2ART(BaseEstimator, ClusterMixin):
    """L2ART.

    Parameters
    ----------
    max_iter : int
        Maximum number of iterations of the ART2 algorithm to run.
    max_class : int
        Maximum number of the class to classify.
    rho : float, default=0.75
        Threshold for degree of match.
    alpha : float, default=1e-5
        Choice parameter.
    beta : float, default=0.1
        Learning rate.
    """

    def __init__(
            self,
            max_iter=10,
            max_class=100,
            rho=0.75,
            alpha=1e-5,
            beta=0.1):
        super().__init__()
        self.max_iter = max_iter
        self.max_class = max_class
        self.rho = rho
        self.alpha = alpha
        self.beta = beta

        self._big_m = 10e10

    def _check_params(self, X):
        # max_iter
        if self.max_iter <= 0:
            raise ValueError(
                f"max_iter must be > 0, got {self.max_iter} instead.")
        # max_class
        if self.max_class <= 0:
            raise ValueError(
                f"max_class must be > 0, got {self.max_class} instead.")
        # rho
        if not (0.0 < self.rho < 1.0):
            raise ValueError(
                f"rho must be in range (0, 1), got {self.rho} instead.")
        # alpha
        if self.alpha <= 0:
            raise ValueError(
                f"alpha must be > 0, got {self.alpha} instead.")
        # beta
        if not (0.0 < self.beta < 1.0):
            raise ValueError(
                f"beta must be in range (0, 1), got {self.beta} instead.")

    def _initialize(self, X):
        self.history_ = {"gap": [], "similarity": []}
        self._n_features = X.shape[1]
        self._w = self._big_m * np.ones((self.max_class, self._n_features))
        self._active_list = [False] * self.max_class

    def fit(self, X):
        """Compute L2ART clustering.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.

        Returns
        -------
        self
            Fitted estimator.
        """
        X = check_array(X)
        self._check_params(X)
        self._initialize(X)
        for _ in range(self.max_iter):
            y = self._resonance_iter(X)
        self.labels_ = y
        return self

    def partial_fit(self, X):
        check_is_fitted(self)
        self._resonance_iter(X)
        return self

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
            Returns -1 label if the match class does not exist.
        """
        check_is_fitted(self)
        X = check_array(X)
        return self._resonance_iter(X, should_update=False)

    def get_center(self):
        return self._w[self._active_list]

    def _resonance_iter(self, X, should_update=True):
        y = -1 * np.ones(X.shape[0], dtype=np.int32)
        for idx, s in enumerate(X):
            # F1
            T = np.linalg.norm(s - self._w, axis=1)

            # Search match
            for _ in range(self.max_class):
                # F2 : Code selection
                J = np.argmin(T)

                # Match
                if (not self._active_list[J]) or (T[J] <= self.rho):
                    if should_update:
                        y[idx] = J

                        # Updata parameters
                        w_previous = self._w.copy()
                        if self._active_list[J]:
                            self._w[J] = self.beta * s + \
                                (1 - self.beta) * self._w[J]
                        else:
                            self._w[J] = s

                        gap = np.mean(np.abs(w_previous - self._w))
                        # logging
                        self.history_["gap"].append(gap)
                        self.history_["similarity"].append(T)

                        self._active_list[J] = True

                    else:
                        if self._active_list[J]:
                            y[idx] = J
                    break

                # Do not match
                else:
                    T[J] = np.max(T) + 1
                    continue
        return y
