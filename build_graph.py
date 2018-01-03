import numpy as np
from scipy.spatial.distance import pdist, squareform


def exponential_euclidian(X, sigma2, cosine=False):
    """
    this function compute the similarity graph using the exponential
    euclidian similarity
    input : matrix X of size n x d. n number of vector of dimension d
    output a complete similarity graph of size n x n
    """
    if(cosine):
        similarities = squareform(pdist(X, 'cosine'))
    else:
        similarities = squareform(pdist(X, 'sqeuclidean'))
    
    return np.exp(-similarities/(2*sigma2))


def build_similarity_graph(X, sigma2, graph_type='knn', graph_thresh=10,
                           cosine=False):
    """
    this function use the a similarity distance to
    build either a knn graph or a epsilon graph
    input : matrix X of size n x d. n number of vectors of dimension d
    output : W size n x n. the similarity graph.
    """
    
    similarities = exponential_euclidian(X, sigma2, cosine)
    W = np.zeros(similarities.shape)
    if(graph_type == 'knn'):
        # j_index : line of the maximums
        # i_index : column of the maximums
        # z_index : the maximums themselves
        ind = np.argsort(similarities)[..., ::-1]  # sort the similarities
        i_index = ind[:, 0:graph_thresh]
        j_index = np.tile(np.arange(W.shape[0]),
                          (W.shape[0], 1)).T[:, 0:graph_thresh]
        z_index = np.sort(similarities)[..., ::-1]
        z_index = z_index[:, 0:graph_thresh]
        W[j_index.ravel(), i_index.ravel()] = z_index.ravel()
        W[i_index.ravel(), j_index.ravel()] = z_index.ravel()
        
    elif (graph_type == 'eps'):
        W[similarities >= graph_thresh] = similarities[
            similarities >= graph_thresh]

    else:
        raise ValueError("Wrong argument returned None")
        
    return W
    



