
import numpy as np

def reconstruct3D(transform_candidates, calibrated_1, calibrated_2):
    best_num_front = -1
    best_candidate = None
    best_lambdas = None
    
    for candidate in transform_candidates:
        R = candidate['R'] 
        T = candidate['T']

        lambdas = np.zeros((2, calibrated_1.shape[0]))
        A = np.zeros((3, 2))
        b = np.zeros((3, 1))

        for i in range(calibrated_1.shape[0]):
            x1, x2 = calibrated_1[i], calibrated_2[i]
            
            A[:, 0] = x2
            A[:, 1] = -R @ x1
            b = T.reshape(-1, 1)

            lambdas[:, i] = np.linalg.lstsq(A, b, rcond=None)[0].squeeze()

        num_front = np.sum(np.logical_and(lambdas[0] > 0, lambdas[1] > 0))

        if num_front > best_num_front:
            best_num_front = num_front
            best_candidate = candidate  
            best_lambdas = lambdas

    P1 = best_lambdas[1].reshape(-1, 1) * calibrated_1
    P2 = best_lambdas[0].reshape(-1, 1) * calibrated_2
    T = best_candidate['T']
    R = best_candidate['R']
    
    return P1, P2, T, R