import numpy as np
from build_laplacian import compute_eig, nystrom_extension


def purity(C, C_hat):
    """
    compute the purity of two sets(matrix of N x num_class)
    """
    # DONE: Compute the purity of two sets
    lab = np.argmax(C, axis=1)
    lab_hat = np.argmax(C_hat, axis=1)
    
    return np.sum(lab == lab_hat) / lab.shape[0]


def semi_supervised(W, mask, label, num_class, m, dt, mu, delta,
                    pure=0.99, nyst=True,
                    lap_type='sym', verb=True):
    """
    this algorithm performs a segmentation of an image using a SSL.
    
    Input : W a graph, num_class the number of classes, m
    is the number of computed eigenvectors, dt
    dt, time step for the heat equation, mu
    a confidence of the data term, delta is the threshold
    mask : which pixel is labeled. a N size vector
    label : the different label. a vector that contains the known
    label for each pixel. the position is known thanks to mask
    """
    # Eigen Value And Eigenvector
    if(nyst):
        S, V = nystrom_extension(W, m, lap_type)
    else:
        S, V = compute_eig(W, m, lap_type)

    n = 1
    if(verb):
        print('Computation of eigenvalues : Done')

    # Constants
    N = W.shape[0]
    row = np.arange(N)

    # 1 . Initialisation
    # u will be the label for each class
    # TODO : Change the initialisation
    u_beg = np.random.randint(0, num_class, N)
    u_beg[np.where(mask == 1)] = label
    u = np.eye(num_class)[u_beg]
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
    if(verb):
        print('computation of one iteration : Done')
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
        if(verb):
            print('iteration '+str(n-1))

    print('Done')
    return u1
        
    
    
    
    
    
