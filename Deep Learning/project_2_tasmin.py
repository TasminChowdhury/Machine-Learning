
# coding: utf-8

# In[1]:

import numpy as np
def zca_whitening_matrix(X):
    #INPUT:  X: [M x N] matrix.
     #   Rows: Variables
     #   Columns: Observations
    #OUTPUT: ZCAMatrix: [M x N] matrix

    # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
    sigma = np.cov(X, rowvar=True) # [M x M]
    # Singular Value Decomposition. X = U * np.diag(S) * V
    U,S,V = np.linalg.svd(sigma)
        # U: [M x M] eigenvectors of sigma.
        # S: [M x 1] eigenvalues of sigma.
        # V: [M x M] transpose of U
    # Whitening constant: prevents division by zero
    epsilon = 1e-5
    # ZCA Whitening matrix: U * Lambda * U'
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T)) # [M x M]
    return ZCAMatrix
#And an example of the usage:

X = np.array([[10, 12, 12], [11, 11, 10], [21, 10, 11], [11, 13, 51], [1, 1, 1] ]) # Input: X [5 x 3] matrix
ZCAMatrix = zca_whitening_matrix(X) # get ZCAMatrix
ZCAMatrix # [5 x 5] matrix
xZCAMatrix = np.dot(ZCAMatrix, X) # project X onto the ZCAMatrix
xZCAMatrix # [5 x 3] matrix


# In[2]:

#I constructed a covariance matrix, then I performed eigen decomposition on the matrix, 
#covariance matrix is symmetric, it describes variance among the data and covariance among the variable. 
#covariance value says how two variable change with respect to each other. 
#performing eigen decomposition on covariance matrix, helps me to find the hidden forces in data
#Then after having the eigen pair, U and S. now I can determine the principle component and select them.
#those which have less eigen value we can drop them. 
#Then I performed ZCA whitening which is simply a linear transform , it takes the input
#and preprocess with this equation.
#After this preprocessing, now we get better feature representation of our data which shows in the result matrix


# In[ ]:



