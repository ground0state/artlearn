import numpy as np
from artlearn import SFAM


def prepare_data():
    X = np.array([[0.30337076, 0.6262552],
                  [0.97010031, 0.],
                  [0.41828834, 0.28989424],
                  [0.14816287, 0.64219202],
                  [0.02348559, 0.94155722],
                  [0.37588878, 0.31753854],
                  [0.61530313, 0.32336765],
                  [0.38154371, 0.34569914],
                  [0., 1.],
                  [0.99509549, 0.08014449],
                  [0.12803322, 0.55447042],
                  [0.29659682, 0.5670909],
                  [0.70892053, 0.28030269],
                  [0.96172813, 0.04089896],
                  [0.00478399, 0.8922326],
                  [0.04425634, 0.47952031],
                  [0.65274233, 0.30290336],
                  [0.01833742, 0.39591246],
                  [0.01001124, 0.42384553],
                  [0.65761849, 0.21432762],
                  [0.49107019, 0.25441864],
                  [0.96370333, 0.04074017],
                  [1., 0.07387525],
                  [0.31979732, 0.61569525],
                  [0.06197319, 0.88100844],
                  [0.04199175, 0.92210155],
                  [0.35776174, 0.59941528],
                  [0.73749139, 0.30674069],
                  [0.45290202, 0.40005298],
                  [0.04629763, 0.50236632]])
    y = np.array([0, 4, 1, 0, 3, 1, 5, 1, 3, 4, 2, 0, 5, 4, 3, 2, 5, 2, 2, 5, 1, 4,
                  4, 0, 3, 3, 0, 5, 1, 2])
    X_ = X.copy()
    X_ = np.hstack([X, 1.0 - X_])
    return X_, y


def test_fit():
    X, y = prepare_data()
    y_answer = np.array([0, 4, 1, 0, 3, 1, 5, 1, 3, 4, 2, 0, 5, 4, 3, 2, 5, 2, 2, 5, 1, 4,
                         4, 0, 3, 3, 0, 5, 1, 2], dtype=np.int32)

    clf = SFAM(max_iter=100, max_class=100, rho=0.9, alpha=1e-5, beta=0.1)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert np.all(y_pred == y_answer)

    X_new = np.array([[0.4, 0.3, 0.6, 0.7]])
    y_pred = clf.predict(X_new)
    y_answer_2 = np.array([1], dtype=np.int32)
    assert np.all(y_pred == y_answer_2)
