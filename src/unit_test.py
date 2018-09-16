import numpy as np
import scipy.linalg
from timeit import default_timer as timer
import pysvd3 as svd3

N = 100000
def get_func(method):
    if method == 'qr':
        func = svd3.qr3
        stat_func = lambda a, o: (a, *o, np.matmul(o[0], o[1]), \
                np.linalg.norm(np.matmul(o[0], o[1]) - a)/\
                np.linalg.norm(a))
        data_func = lambda x: (np.random.randn(x, 3, 3).astype(np.float32),)
        print_fmt = 'matrix A: \n {}\nmatrix Q:\n {}\nmatrix R:\n {}\n' \
                + 'matrix QR:\n {}\nerror: {}'

    elif method == 'np_qr':
        func = np.linalg.qr
        stat_func = lambda a, o: (a, *o, np.matmul(o[0], o[1]), \
                np.linalg.norm(np.matmul(o[0], o[1]) - a)/\
                np.linalg.norm(a))
        data_func = lambda x: (np.random.randn(x, 3, 3).astype(np.float32),)
        print_fmt = 'matrix A: \n {}\nmatrix Q:\n {}\nmatrix R:\n {}\n' \
                + 'matrix QR:\n {}\nerror: {}'

    elif method == 'svd':
        func = svd3.svd3
        stat_func = lambda a, o: (a, *o, np.matmul(o[0]*o[1], o[2]), \
                np.linalg.norm(np.matmul(o[0]*o[1], o[2]) - a)/\
                np.linalg.norm(a))
        data_func = lambda x: (np.random.randn(x, 3, 3).astype(np.float32),)
        print_fmt = 'matrix A: \n {}\nmatrix U:\n {}\nmatrix S:\n {}\n' \
                + 'matrix Vh:\n {}\nmatrix USV*:\n {}\nerror: {}'

    elif method == 'np_svd':
        func = np.linalg.svd
        stat_func = lambda a, o: (a, *o, np.matmul(o[0]*o[1], o[2]), \
                np.linalg.norm(np.matmul(o[0]*o[1], o[2]) - a)/\
                np.linalg.norm(a))
        data_func = lambda x: (np.random.randn(x, 3, 3).astype(np.float32),)
        print_fmt = 'matrix A: \n {}\nmatrix U:\n {}\nmatrix S:\n {}\n' \
                + 'matrix Vh:\n {}\nmatrix USV*:\n {}\nerror: {}'

    elif method == 'pd':
        func = svd3.pd3
        stat_func = lambda a, o: (a, *o, np.matmul(o[0], o[1]), \
                np.linalg.norm(np.matmul(o[0], o[1]) - a)/\
                np.linalg.norm(a))
        data_func = lambda x: (np.random.randn(x, 3, 3).astype(np.float32),)
        print_fmt = 'matrix A: \n {}\nmatrix U:\n {}\nmatrix P:\n {}\n' \
                + 'matrix UP:\n {}\nerror: {}'

    elif method == 'np_pd':
        func = scipy.linalg.polar
        stat_func = lambda a, o: (a, *o, np.matmul(o[0], o[1]), \
                np.linalg.norm(np.matmul(o[0], o[1]) - a)/\
                np.linalg.norm(a))
        data_func = lambda x: (np.random.randn(x, 3, 3).astype(np.float32),)
        print_fmt = 'matrix A: \n {}\nmatrix U:\n {}\nmatrix P:\n {}\n' \
                + 'matrix UP:\n {}\nerror: {}'

    else:
        error('Unknown method: {}'.format(method))

    return data_func, func, stat_func, print_fmt

def test_correctness(data_func, func, stat_func, print_fmt):

    datas = data_func(1)
    for data in zip(*datas):
        out = stat_func(*data, func(*data))
        print(print_fmt.format(*out))
    return


def benchmark_accuracy(data_func, func, stat_func):

    datas = data_func(N)
    err_sum = 0
    for data in zip(*datas):
        err_sum += stat_func(*data, func(*data))[-1]

    print('Average error over {} random samples: {}'.format(N, err_sum/N))

    return


def benchmark_speed(data_func, func):

    datas = data_func(N)

    start = timer()
    for data in zip(*datas):
        _ = func(*data)
    end = timer()

    print('Average execution time over {} random samples: {} us'.format(N, (end-start)/N*1e6))

    return


def test_method_correctness(method):
    print('Testing method: {}'.format(method))
    funcs = get_func(method)
    test_correctness(*funcs)
    return


def benchmark_method_accuracy(method):
    print('Benchmarking accuracy: {}'.format(method))
    funcs = get_func(method)
    benchmark_accuracy(*funcs[:-1])
    return


def benchmark_method_speed(method):
    print('Benchmarking speed: {}'.format(method))
    funcs = get_func(method)
    benchmark_speed(*funcs[:-2])
    return


if  __name__ == '__main__':

    #test_method_correctness('qr')
    #test_method_correctness('svd')
    #test_method_correctness('pd')
    benchmark_method_accuracy('qr')
    benchmark_method_accuracy('np_qr')
    benchmark_method_accuracy('svd')
    benchmark_method_accuracy('np_svd')
    benchmark_method_accuracy('pd')
    benchmark_method_accuracy('np_pd')
    benchmark_method_speed('qr')
    benchmark_method_speed('np_qr')
    benchmark_method_speed('svd')
    benchmark_method_speed('np_svd')
    benchmark_method_speed('pd')
    benchmark_method_speed('np_pd')
