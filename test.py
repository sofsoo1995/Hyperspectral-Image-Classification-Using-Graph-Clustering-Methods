import numpy as np
from build_laplacian import nystrom_extension, compute_eig
from build_graph import build_similarity_graph
from segmentation import semi_supervised
from spectral import open_image
import time


X = np.random.randn(100, 2)
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

print('read an image')

print('test semi supervised')

A = open_image('../database/Urban_F210/Urban_F210.hdr').load()
A = A.reshape(A.shape[0]*A.shape[1], A.shape[2])
start = time.time()
print('Build Graph')
W = build_similarity_graph(A, 10, 'eps', 0)
print('Done')
print(time.time()-start)
start = time.time()
print('Compute nystrom ext')
S, V = nystrom_extension(W, 10)
print('Done')
print(time.time()-start)
print(S)

