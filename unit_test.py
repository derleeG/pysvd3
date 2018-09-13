import numpy as np
from svd3 import svd3 as svd


def test_svd_correctness():

    A = np.random.rand(3, 3).astype(np.float32)
    U, S, V = svd(A)
    S = np.diag(np.diag(S))
    print('matrix A:')
    print(A)
    print('matrix U:')
    print(U)
    print('matrix S:')
    print(S)
    print('matrix V:')
    print(V)
    print('matrix USVt')
    print(np.matmul(np.matmul(U, S), V.T))





if  __name__ == '__main__':



    test_svd_correctness()


