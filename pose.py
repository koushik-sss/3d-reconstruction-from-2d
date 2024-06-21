import numpy as np
import numpy as np

def pose_candidates_from_E(E):
    transform_candidates = []
    U, _, VT = np.linalg.svd(E)
    
    # Case 1: T from U, R = URz(π/2)^T V^T
    T = U[:, 2]
    R = U @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).T @ VT
    transform_candidates.append({'T': T, 'R': R})
    
    # Case 2: T from U, R = URz(-π/2)^T V^T
    T = U[:, 2]
    R = U @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]).T @ VT
    transform_candidates.append({'T': T, 'R': R})
    
    # Case 3: T from -U, R = URz(π/2)^T V^T
    T = -U[:, 2]
    R = U @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).T @ VT
    transform_candidates.append({'T': T, 'R': R})
    
    # Case 4: T from -U, R = URz(-π/2)^T V^T
    T = -U[:, 2]
    R = U @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]).T @ VT
    transform_candidates.append({'T': T, 'R': R})
    
    return transform_candidates