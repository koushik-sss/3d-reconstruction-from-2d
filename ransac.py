from lse import least_squares_estimation
import numpy as np
def ransac_estimator(X1, X2, num_iterations=60000):
    sample_size = 8
    eps = 10**-4
    
    best_num_inliers = -1
    best_inliers = None
    best_E = None
    
    for i in range(num_iterations):
        permuted_indices = np.random.RandomState(seed=(i*10)).permutation(np.arange(X1.shape[0]))
        sample_indices = permuted_indices[:sample_size]
        test_indices = permuted_indices[sample_size:]
        
        E = least_squares_estimation(X1[sample_indices], X2[sample_indices])
        
        residuals = []
        for j in test_indices:
            x1, x2 = X1[j], X2[j]
            epipolar_line = E.dot(x1)
            residual = (x2.dot(epipolar_line))**2 / (epipolar_line[0]**2 + epipolar_line[1]**2)
            residual += (x1.dot(E.T.dot(x2)))**2 / (E.T.dot(x2)[0]**2 + E.T.dot(x2)[1]**2)
            residuals.append(residual)

        residuals = np.array(residuals) 
        inliers = test_indices[residuals < eps]
        
        if inliers.shape[0] > best_num_inliers:
            best_num_inliers = inliers.shape[0]
            best_E = E
            best_inliers = np.concatenate((sample_indices, inliers))

    return best_E, best_inliers