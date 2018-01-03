import numpy as np
from build_laplacian import nystrom_extension


def purity(C, C_hat):
    """
    compute the purity of twoy sets(matrix of N x num_class)
    """
    # DONE: Compute the purity of two sets
    lab = np.argmax(C, axis=1)
    lab_hat = np.argmax(C_hat, axis=1)
    return np.sum(lab == lab_hat) / lab.shape[0]


def semi_supervised(X, sigma2, label, num_class, m, dt, mu,
                    pure=0.99, cosine=True,
                    lap_type='sym', verb=True):
    """
    this algorithm performs a segmentation of an image using a SSL.
    
    Input : W a graph, num_class the number of classes, m
    is the number of computed eigenvectors, dt
    dt, time step for the heat equation, mu
    a confidence of the data term, delta is the threshold
    label : the class known by the pixel -1 if not
    
    """
    # Eigen Value And Eigenvector

    S, V = nystrom_extension(X, m, sigma2, cosine=True)  # Const
    u0 = np.random.randint(0, num_class, X.shape[0])
    u0[label >= 0] = label[label >= 0]
    u0 = np.eye(num_class)[u0]
    u_hat = u0  # Const
    lamb = np.tile((label >= 0), (num_class, 1)).T  # Const
    d0 = V.T.dot(mu*lamb*(u0-u_hat))
    u = u0
    d = d0
    n = 0
    is_pure = True
    
    while(is_pure):
    
        a = V.T.dot(u)
  
        for k in range(m):
            a[k] = (1-dt*S[k])*a[k]-dt*d[k]
        u1 = V.dot(a)
        d = mu*V.T.dot(lamb*(u1-u_hat))
    
        r = np.argmax(u1, axis=1)
        u1 = np.eye(num_class)[r]
        n = n+1
        print(str(n)+' '+str(purity(u, u1)))
        if(purity(u, u1) > pure):
            is_pure = False
        u = u1
    return u1
        
    
    
    
    
    
