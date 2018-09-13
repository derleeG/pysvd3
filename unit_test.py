import numpy as np
from svd3 import svd3 as svd


def test_svd_correctness():

    A = np.random.rand(3, 3).astype(np.float32)
    print('fast svd3 implementation')
    U, S, Vh = svd(A)
    print('matrix A:')
    print(A)
    print('matrix U:')
    print(U)
    print('matrix S:')
    print(S)
    print('matrix Vh:')
    print(Vh)
    print('matrix USVt:')
    USVt = np.matmul(U * S, Vh)
    print(USVt)
    print('USVt - A')
    print(USVt - A)

    print('np.linalg.svd implementation')
    U, S, Vh = np.linalg.svd(A)
    print('matrix A:')
    print(A)
    print('matrix U:')
    print(U)
    print('matrix S:')
    print(S)
    print('matrix Vh:')
    print(Vh)
    print('matrix USVt:')
    USVt = np.matmul(U * S, Vh)
    print(USVt)
    print('USVt - A')
    print(USVt - A)


def benchmark_accuracy():

    N = 100000
    a = np.random.rand(N, 3, 3).astype(np.float32)

    err_sum = 0
    for A in a:
        U, S, Vh = svd(A)
        USVt = np.matmul(U * S, Vh)
        err_sum += np.linalg.norm(USVt - A)

    print('fast svd3')
    print('Average error (||USV* - A||F) over {} random samples: {}'.format(N, err_sum/N))

    err_sum = 0
    for A in a:
        U, S, Vh = np.linalg.svd(A)
        USVt = np.matmul(U * S, Vh)
        err_sum += np.linalg.norm(USVt - A)

    print('np linalg')
    print('Average error (||USV* - A||F) over {} random samples: {}'.format(N, err_sum/N))


def benchmark_speed():

    from timeit import default_timer as timer

    N = 100000
    a = np.random.rand(N, 3, 3).astype(np.float32)

    start = timer()
    for A in a:
        _, _, _ = svd(A)
    end = timer()

    print('fast svd3')
    print('Average execution time over {} random samples: {} ms'.format(N, (end-start)/N*1e6))

    start = timer()
    for A in a:
        _, _, _ = np.linalg.svd(A)
    end = timer()

    print('np linalg')
    print('Average execution time over {} random samples: {} ms'.format(N, (end-start)/N*1e6))


if  __name__ == '__main__':

    #test_svd_correctness()
    benchmark_accuracy()
    benchmark_speed()
