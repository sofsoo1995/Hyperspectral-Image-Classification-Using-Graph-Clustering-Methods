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
