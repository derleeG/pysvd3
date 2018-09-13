import numpy as np
from timeit import default_timer as timer
from svd3 import svd3 as svd
from svd3 import qr3 as qr
from svd3 import pd3 as pd



def get_func(method):
    if method == 'qr':
        func = qr
        stat_func = lambda a, q, r: (a, q, r, np.matmul(q, r), \
                np.linalg.norm(np.matmul(q, r) - a))
        print_fmt = 'matrix A: \n {}\nmatrix Q:\n {}\nmatrix R:\n {}\n' \
                + 'matrix QR:\n {}\nerror: {}'

        return func, stat_func, print_fmt

    elif method == 'svd':
        func = svd
        stat_func = lambda a, u, s, vh: (a, u, s, vh, np.matmul(u*s, vh), \
                np.linalg.norm(np.matmul(u*s, vh) - a))
        print_fmt = 'matrix A: \n {}\nmatrix U:\n {}\nmatrix S:\n {}\n' \
                + 'matrix Vh:\n {}\nmatrix USV*:\n {}\nerror: {}'

        return func, stat_func, print_fmt

    elif method == 'pd':
        func = pd
        stat_func = lambda a, u, p: (a, u, p, np.matmul(u, p), \
                np.linalg.norm(np.matmul(u, p) - a))
        print_fmt = 'matrix A: \n {}\nmatrix U:\n {}\nmatrix P:\n {}\n' \
                + 'matrix UP:\n {}\nerror: {}'

        return func, stat_func, print_fmt


def test_correctness(func, stat_func, print_fmt, data=None):
    if data is None:
        data = np.random.rand(3, 3).astype(np.float32)

    out = func(data)
    out = stat_func(data, *out)
    print(print_fmt.format(*out))

    return data


def benchmark_accuracy(func, stat_func, data=None):
    if data is None:
        N = 100000
        data = np.random.rand(N, 3, 3).astype(np.float32)
    else:
        N = data.shape[0]

    err_sum = 0
    for A in data:
        out = func(A)
        out = stat_func(A, *out)
        err_sum += out[-1]

    print('Average error over {} random samples: {}'.format(N, err_sum/N))

    return data


def benchmark_speed(func, data=None):
    if data is None:
        N = 100000
        data = np.random.rand(N, 3, 3).astype(np.float32)
    else:
        N = data.shape[0]

    start = timer()
    for A in data:
        _ = func(A)
    end = timer()

    print('Average execution time over {} random samples: {} ms'.format(N, (end-start)/N*1e6))

    return data


def test_method_correctness(method, data=None):
    print('Testing method: {}'.format(method))
    funcs = get_func(method)
    return test_correctness(*funcs, data)


def benchmark_method_accuracy(method, data=None):
    print('Benchmarking accuracy: {}'.format(method))
    funcs = get_func(method)
    return benchmark_accuracy(*funcs[:-1], data)


def benchmark_method_speed(method, data=None):
    print('Benchmarking speed: {}'.format(method))
    funcs = get_func(method)
    return benchmark_speed(funcs[0], data)



if  __name__ == '__main__':

    test_method_correctness('qr')
    test_method_correctness('svd')
    test_method_correctness('pd')
    benchmark_method_accuracy('qr')
    benchmark_method_accuracy('svd')
    benchmark_method_accuracy('pd')
    benchmark_method_speed('qr')
    benchmark_method_speed('svd')
    benchmark_method_speed('pd')
