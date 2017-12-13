import numpy as np
from sklearn.cluster import KMeans


def build_laplacian(W, lap_type='sym'):
    """
    this function build the laplacian of a graph
    input : W a matrix that represent the similarity graph
    W is a n x n matrix.
    output L is the laplacian of the graph represented by W
    """
    degree = np.sum(W, axis=1)
    L = np.zeros(W.shape)
    if(lap_type == 'unn'):
        D = np.diag(degree)
        L = D-W
        
    elif(lap_type == 'sym'):
        degree_sq_inv = 1/np.sqrt(degree)
        D = np.diag(degree_sq_inv)
        L = np.eye(W.shape[0]) - D.dot(W.dot(D))

    elif(lap_type == 'rw'):
        degree_inv = 1/degree
        D = np.diag(degree_inv)
        L = np.eye(W.shape[0]) - D.dot(W)
    else:
        raise ValueError('unknown type of laplacian')

    return L


def compute_eig(W, m, lap_type='sym'):
    """
    this function compute the m eigenvector of the laplacian of W
    input : W a n x n matrix the matrix of the graph
            m the number of eigenvector we keep
    output : the label of each point and the matrix that contains the
             K first eig
    """
    L = build_laplacian(W, lap_type)
    S, V = np.linalg.eig(L)
    ind = np.argsort(S)
    S = S[ind]
    V = V[:, ind]
    V = V[:, 0:m]
    S = S[0:m]
    return S, V
    

def spectral_clustering(W, K, lap_type='sym'):
    """
    this algorithm perform the spectral clustering
    input : W a n x n matrix of the graph
            K the number of cluster
    output : label of each point
    """
    S, V = compute_eig(W, K, lap_type)
    kmeans = KMeans(n_clusters=K)
    label = kmeans.fit_predict(V)
    return label


def nystrom_extension(W, m, lap_type='sym'):
    '''
    this function compute the eigenvector and the eigenvalues of a
    matrix using the nystrom extension.
    See Part 3.3 of the article
    input : the matrix W size n x n. the graph m the number of
    node we choose randomly.
    output : an approximation of the eigenvalues and the eigenfunction
    S and Phi
    '''
    set_Z = np.random.permutation(W.shape[0])
    subset_X = set_Z[0:m]
    subset_Y = set_Z[m:]
    
    # selection of the subgraph
    Wxx = W[np.ix_(subset_X, subset_X)]
    Wxy = W[np.ix_(subset_X, subset_Y)]
    Wyx = W[np.ix_(subset_Y, subset_X)]
    
    Wxx_inv = np.linalg.pinv(Wxx)

    # normalized versions of Wxx and Wxy
    dx = Wxx.dot(np.ones((Wxx.shape[0], 1))) + Wxy.dot(
        np.ones((Wxy.shape[1], 1)))

    dy = Wyx.dot(np.ones((Wyx.shape[1], 1))) + Wyx.dot(
        Wxx_inv).dot(Wxy).dot(np.ones((Wyx.shape[0], 1)))

    sx = np.sqrt(dx)
    sy = np.sqrt(dy)
    
    Wxx = Wxx / sx.dot(sx.T)
    Wxy = Wxy / sx.dot(sy.T)
    Wyx = Wxy.T

    # computation of eigenvalues approx

    Gamma, B = np.linalg.eig(Wxx)
    Wxx_mdemi = B.dot(np.diag(1/np.sqrt(Gamma))).dot(B.T)
    # Mat is the second matrix to diagonalize
    Mat = Wxx + Wxx_mdemi.dot(Wxy.dot(Wyx)).dot(Wxx_mdemi)
    Xi, At = np.linalg.eig(Mat)

    Gamma_demi = np.diag(np.sqrt(Gamma))
    Gamma_mdemi = np.diag(1/np.sqrt(Gamma))
    Xi_mdemi = np.diag(1/np.sqrt(Xi))
    
    Phi1 = B.dot(Gamma_demi.dot(B.T)).dot(At.T.dot(Xi_mdemi))
    Phi2 = Wyx.dot(B.dot(Gamma_mdemi).dot(B.T)).dot(At.T.dot(Xi_mdemi))
    
    Phi = np.vstack((Phi1, Phi2))
    S = 1 - Xi
    return S, Phi
