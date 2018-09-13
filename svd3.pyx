# Cython interface for wrapping svd3 c-library

import numpy as np
cimport numpy as np
np.import_array()


cdef extern from 'lib/svd3/svd3.h':
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


def svd3(np.ndarray[float, ndim=2, mode='c'] A not None):
    assert A.shape[0] == 3 and A.shape[1] == 3

    # prepare output buffer
    U = np.zeros((3, 3), dtype=float)
    S = np.zeros((3, 3), dtype=float)
    V = np.zeros((3, 3), dtype=float)

    cdef float u11, u12, u13, u21, u22, u23, u31, u32, u33
    cdef float s11, s12, s13, s21, s22, s23, s31, s32, s33
    cdef float v11, v12, v13, v21, v22, v23, v31, v32, v33

    svd(A[0, 0], A[0, 1], A[0, 2],
        A[1, 0], A[1, 1], A[1, 2],
        A[2, 0], A[2, 1], A[2, 2],
        u11, u12, u13,
        u21, u22, u23,
        u31, u32, u33,
        s11, s12, s13,
        s21, s22, s23,
        s31, s32, s33,
        v11, v12, v13,
        v21, v22, v23,
        v31, v32, v33)

    U = np.array([[u11, u12, u13], [u21, u22, u23], [u31, u32, u33]])
    S = np.array([[s11, s12, s13], [s21, s22, s23], [s31, s32, s33]])
    #S = np.array([s11, s22, s33])
    V = np.array([[v11, v12, v13], [v21, v22, v23], [v31, v32, v33]])

    return U, S, V



