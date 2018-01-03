import numpy as np
from sklearn.cluster import KMeans
from build_graph import exponential_euclidian
import scipy.spatial.distance as sc
import scipy 

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
    return np.real(S), V
    

def nystrom_extension(X, m, sigma2, cosine=False, lap_type='sym'):
    '''
    this function compute the eigenvector and the eigenvalues of a
    matrix using the nystrom extension.
    Important : the graph is not built entirely.
    See Part 3.3 of the article
    input : the matrix X size n x p. the graph m the number of
    node we choose randomly.
    output : an approximation of the eigenvalues and the eigenfunction
    S and Phi
    Assumption : It is the approximation for the symetrized laplacian.
    for another laplacian, it might not work
    '''
    set_Z = np.random.permutation(X.shape[0])
    subset_X = set_Z[0:m]
    subset_Y = set_Z[m:]

    # selection of the subgraph
    Xm = X[subset_X]
    Xnm = X[subset_Y]
    if(cosine):
        simXY = sc.cdist(Xm, Xnm, 'cosine', np.float64)
        simXX = sc.cdist(Xm, Xm, 'cosine', np.float64)
    else:
        simXX = sc.cdist(Xm, Xm, 'sceuclidean', np.float64)
        simXY = sc.cdist(Xm, Xnm, 'sceuclidean', np.float64)
    Wxx = np.exp(-simXX/(2*sigma2))
    Wxy = np.exp(-simXY/(2*sigma2))
    Wyx = Wxy.T

    # Normalisation of Wxx and Wxy
    d1 = np.sum(np.vstack((Wxx, Wyx)), axis=0)
    d2 = np.sum(Wxy, axis=0) + np.sum(Wyx, axis=0).dot(
        np.linalg.pinv(Wxx).dot(Wxy))
    dhat = np.sqrt(1 / np.hstack((d1, d2)))
    sx = np.array([dhat[0:m]])
    sy = np.array([dhat[m:]])
    
    Wxx = Wxx * (sx.T.dot(sx))
    Wxy = Wxy * (sx.T.dot(sy))
    Wyx = Wxy.T

    # Estimation of eigen vectors in V
    Wxx_si = scipy.linalg.sqrtm(np.linalg.pinv(Wxx))
    Q = Wxx + Wxx_si.dot(Wxy.dot(Wyx)).dot(Wxx_si)
    U, L, Vt = np.linalg.svd(Q)
    V = np.vstack((Wxx, Wyx)).dot(Wxx_si).dot(
        U.dot(np.linalg.pinv(np.sqrt(np.diag(L)))))
    
    # sq_sum = np.sqrt(np.sum(np.multiply(V, V), axis=1))+1e-20
    # sq_sum_mask = np.zeros((len(sq_sum), m), dtype=np.float64)
    # for k in range(m):
    #    sq_sum_mask[:, k] = sq_sum

    # Umat = np.divide(V, sq_sum_mask)
    Umat = V
    Xres = np.zeros((Umat.shape[0], Umat.shape[1]))
    Xres[set_Z, :] = Umat
    return (1-L), Xres


def spectral_clustering(W, m, K, lap_type='sym'):
    """
    this algorithm perform the spectral clustering
    input : W a n x n matrix of the graph
            K the number of cluster
            m the number of eigen vectors
    output : label of each point
    """
    
    S, V = compute_eig(W, m, lap_type)
    kmeans = KMeans(n_clusters=K)
    label = kmeans.fit_predict(V)
    return label
