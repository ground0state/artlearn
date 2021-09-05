import numpy as np
from artlearn import ART1


def prepare_data():
    X = np.array([[1, 1, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 0],
                  [1, 1, 0, 1, 1, 0],
                  [1, 0, 1, 1, 1, 0],
                  [0, 0, 0, 0, 1, 1],
                  [0, 0, 0, 1, 1, 1],
                  [0, 0, 1, 1, 1, 1],
                  [0, 0, 0, 1, 1, 1],
                  [1, 1, 1, 0, 0, 0],
                  [0, 0, 0, 1, 1, 1],
                  [1, 0, 0, 0, 0, 0],
                  [1, 1, 0, 0, 0, 0]])
    y = np.array([0, 1, 1, 1, 2, 2, 2, 2, 0, 2, 0, 0])
    return X, y


def test_fit():
    X, y = prepare_data()
    y_answer = np.array([0, 1, 0, 1, 2, 2, 2, 2, 0, 2, 0, 0], dtype=np.int32)

    clf = ART1(max_iter=10, max_class=5, rho=0.01)
    clf.fit(X)
    assert np.all(clf.labels_ == y_answer)

    y_pred = clf.predict(X)
    assert np.all(y_pred == y_answer)

    X_new = np.array([[1, 1, 0, 0, 0, 0]])
    y_pred = clf.partial_fit_predict(X_new)
    y_answer_2 = np.array([0], dtype=np.int32)
    assert np.all(y_pred == y_answer_2)
