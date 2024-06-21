import numpy as np

def least_squares_estimation(X1, X2):
    num_points = X1.shape[0]
    A = np.zeros((num_points, 9))

    for i in range(num_points):
        x1, y1, w1 = X1[i]
        x2, y2, w2 = X2[i]
        A[i] = [x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, 1]

    _, _, VT = np.linalg.svd(A)
    E = VT[-1].reshape(3, 3)

    U, S, VT = np.linalg.svd(E)
    S = np.diag([1, 1, 0])
    E = np.dot(U, np.dot(S, VT))

    return E