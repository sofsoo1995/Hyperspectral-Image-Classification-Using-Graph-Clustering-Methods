import numpy as np
from build_laplacian import nystrom_extension, compute_eig
from build_graph import build_similarity_graph
import time


X = np.random.randn(1000, 2)
W = build_similarity_graph(X, 1, 'eps', 0)
start = time.time()
Sp, Vp = nystrom_extension(W, 10)
print(time.time()-start)
start = time.time()
S, V = compute_eig(W, 10)
print(time.time()-start)
print(S)
print(Sp)
print(V.shape)
print(Vp.shape)
print(np.linalg.norm(V-Vp)/np.linalg.norm(V))
