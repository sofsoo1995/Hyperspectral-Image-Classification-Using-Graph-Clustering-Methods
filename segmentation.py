import numpy as np
from build_laplacian import compute_eig, nystrom_extension


def purity(C, C_hat):
    """
    compute the purity of two sets(matrix of N x num_class)
    """
    # TODO: Compute the purity of two sets
    return None


def semi_supervised(W, mask, label, num_class, m, dt, mu, delta,
                    pure=0.9999, nyst=True,
                    lap_type='sym'):
    """
    this algorithm performs a segmentation of an image using a SSL.
    
    Input : W a graph, num_class the number of classes, m
    is the number of computed eigenvectors, dt
    dt, time step for the heat equation, mu
    a confidence of the data term, delta is the threshold
    mask : which pixel is labeled. a N size vector
    label : the different label. a matrix that have the same size as u
    """
    # Eigen Value And Eigenvector
    if(nyst):
        S, V = nystrom_extension(W, m, lap_type)
    else:
        S, V = compute_eig(W, m, lap_type)

    n = 1

    # Constants
    N = W.shape[0]
    row = np.arange(N)

    # 1 . Initialisation
    # u will be the label for each class
    # TODO : Change the initialisation
    u = label
    # Computation of d
    d = V.T.dot(mu * mask * (u - label))
    # V.T.dot(V) is the identity

    # 2 . computation of one iteration
    a = V.T.dot(u)
    for k in range(m):
        a[k] = (1-dt*S[k]) - dt*d[k]
    u1 = V.dot(a)
    d = V.T.dot(mu*mask*(u1-label))
    # Compute the maximum label
    r = np.argmax(u1, axis=1)
    mask_r = np.zeros(u1.shape)
    
    mask_r[row, r] = 1
    u1 = u1 * mask_r
    u1[row, r] = 1

    # Iterations.
    while(purity(u, u1) < pure):
        u = u1
        a = V.T.dot(u)
        for k in range(m):
            a[k] = (1-dt*S[k]) - dt*d[k]
        u1 = V.dot(a)
        # update d
        d = V.T.dot(mu*mask*(u1-label))
        # Compute the maximum label
        r = np.argmax(u1, axis=1)
        mask_r = np.zeros(u1.shape)
        mask_r[row, r] = 1
        u1 = u1 * mask_r
        u1[row, r] = 1
        
        n = n+1
    return u1
        
    
    
    
    
    
