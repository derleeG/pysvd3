# Cython interface for wrapping svd3 c-library

import numpy as np
cimport numpy as np
cimport cython
np.import_array()


cdef extern from '../lib/svd3/svd3.h':
    void QRDecomposition(\
            float b11, float b12, float b13, \
            float b21, float b22, float b23, \
            float b31, float b32, float b33, \
            float &q11, float &q12, float &q13, \
            float &q21, float &q22, float &q23, \
            float &q31, float &q32, float &q33, \
            float &r11, float &r12, float &r13, \
            float &r21, float &r22, float &r23, \
            float &r31, float &r32, float &r33)

    void svd(\
            float a11, float a12, float a13, \
            float a21, float a22, float a23, \
            float a31, float a32, float a33, \
            float &u11, float &u12, float &u13, \
            float &u21, float &u22, float &u23, \
            float &u31, float &u32, float &u33, \
            float &s11, float &s12, float &s13, \
            float &s21, float &s22, float &s23, \
            float &s31, float &s32, float &s33, \
            float &v11, float &v12, float &v13, \
            float &v21, float &v22, float &v23, \
            float &v31, float &v32, float &v33)

    void pd(\
            float a11, float a12, float a13, \
            float a21, float a22, float a23, \
            float a31, float a32, float a33, \
            float &u11, float &u12, float &u13, \
            float &u21, float &u22, float &u23, \
            float &u31, float &u32, float &u33, \
            float &p11, float &p12, float &p13, \
            float &p21, float &p22, float &p23, \
            float &p31, float &p32, float &p33)


def qr3(np.ndarray[float, ndim=2, mode='c'] A not None):
    if A.shape[0] != 3 or A.shape[1] != 3:
        raise ValueError("Only 3x3 matrix supported")

    cdef np.ndarray[float, ndim=2, mode='c'] Q = np.zeros((3,3), dtype=np.float32)
    cdef np.ndarray[float, ndim=2, mode='c'] R = np.zeros((3,3), dtype=np.float32)

    QRDecomposition(
        A[0, 0], A[0, 1], A[0, 2],
        A[1, 0], A[1, 1], A[1, 2],
        A[2, 0], A[2, 1], A[2, 2],
        Q[0, 0], Q[0, 1], Q[0, 2],
        Q[1, 0], Q[1, 1], Q[1, 2],
        Q[2, 0], Q[2, 1], Q[2, 2],
        R[0, 0], R[0, 1], R[0, 2],
        R[1, 0], R[1, 1], R[1, 2],
        R[2, 0], R[2, 1], R[2, 2])

    R[1, 0] = 0
    R[2, 0] = 0
    R[2, 1] = 0

    return Q, R


def svd3(np.ndarray[float, ndim=2, mode='c'] A not None):
    if A.shape[0] != 3 or A.shape[1] != 3:
        raise ValueError("Only 3x3 matrix supported")

    cdef np.ndarray[float, ndim=2, mode='c'] U = np.zeros((3,3), dtype=np.float32)
    cdef np.ndarray[float, ndim=2, mode='c'] S = np.zeros((3,3), dtype=np.float32)
    cdef np.ndarray[float, ndim=2, mode='c'] V = np.zeros((3,3), dtype=np.float32)

    svd(A[0, 0], A[0, 1], A[0, 2],
        A[1, 0], A[1, 1], A[1, 2],
        A[2, 0], A[2, 1], A[2, 2],
        U[0, 0], U[0, 1], U[0, 2],
        U[1, 0], U[1, 1], U[1, 2],
        U[2, 0], U[2, 1], U[2, 2],
        S[0, 0], S[1, 1], S[2, 2],
        S[1, 0], S[0, 1], S[1, 2],
        S[2, 0], S[2, 1], S[0, 2],
        V[0, 0], V[1, 0], V[2, 0],
        V[0, 1], V[1, 1], V[2, 1],
        V[0, 2], V[1, 2], V[2, 2])

    return U, S[0, :], V


def pd3(np.ndarray[float, ndim=2, mode='c'] A not None):
    if A.shape[0] != 3 or A.shape[1] != 3:
        raise ValueError("Only 3x3 matrix supported")

    cdef np.ndarray[float, ndim=2, mode='c'] U = np.zeros((3,3), dtype=np.float32)
    cdef np.ndarray[float, ndim=2, mode='c'] P = np.zeros((3,3), dtype=np.float32)

    pd(
        A[0, 0], A[0, 1], A[0, 2],
        A[1, 0], A[1, 1], A[1, 2],
        A[2, 0], A[2, 1], A[2, 2],
        U[0, 0], U[0, 1], U[0, 2],
        U[1, 0], U[1, 1], U[1, 2],
        U[2, 0], U[2, 1], U[2, 2],
        P[0, 0], P[0, 1], P[0, 2],
        P[1, 0], P[1, 1], P[1, 2],
        P[2, 0], P[2, 1], P[2, 2])

    return U, P



