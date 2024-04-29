import cython
from libc.math cimport cos, sin, M_PI, sqrt
import numpy as np
cimport numpy as cnp
cnp.import_array()


cpdef cnp.ndarray[int,ndim=2] init_vol_symm_map_cython(double[:,:] B, double[:,:] invB, cnp.ndarray[int,ndim=2] symm,
                                        double qres, double qlim,
                                        int n, long order, double[:,:,:] table):
    cdef int ix, iy, iz
    cdef int eix, eiy, eiz, eindex
    cdef int index, ord
    cdef double qx, qy, qz
    cdef double eqx, eqy, eqz
    cdef double h, k, l
    cdef double eh, ek, el

    cdef long[:,:] symms = symm

    for ix in range(n):
        qx = ix*qres - qlim

        for iy in range(n):
            qy = iy * qres - qlim

            for iz in range(n):
                qz = iz * qres - qlim
                index = iz + iy*n + ix*n*n

                h = qx * invB[0,0] + qy * invB[0,1] + qz * invB[0,2]
                k = qx * invB[1,0] + qy * invB[1,1] + qz * invB[1,2]
                l = qx * invB[2,0] + qy * invB[2,1] + qz * invB[2,2]

                for ord in range(order):
                    eh = h * table[ord,0,0] + k * table[ord,0,1] + l * table[ord,0,2]
                    ek = h * table[ord,1,0] + k * table[ord,1,1] + l * table[ord,1,2]
                    el = h * table[ord,2,0] + k * table[ord,2,1] + l * table[ord,2,2]

                    eqx = eh * B[0,0] + ek * B[0,1] + el * B[0,2]
                    eqy = eh * B[1,0] + ek * B[1,1] + el * B[1,2]
                    eqz = eh * B[2,0] + ek * B[2,1] + el * B[2,2]

                    eix = round((eqx+qlim)/qres)
                    if eix >= n or eix < 0:
                        eix = -1
                    eiy = round((eqy+qlim)/qres)
                    if eiy >= n or eiy < 0:
                        eiy = -1
                    eiz = round((eqz+qlim)/qres)
                    if eiz >= n or eiz < 0:
                        eiz = -1

                    if eix < 0 or eiy < 0 or eiz < 0:
                        # one of the indices was out of bounds; set to -1 to mean 'no equivalent'
                        symms[index, ord] = -1
                    else:
                        eindex = eiz + eiy*n + eix*n*n
                        symms[index,ord] = eindex
    return symm

cpdef cnp.ndarray[int,ndim=2] appl_vol_sym_cython(cnp.ndarray[int,ndim=1] old_q, cnp.ndarray[int,ndim=1] qspace_flat, long numpix,
                                                 long order, cnp.ndarray[long,ndim=2] symm):
    cdef int pix, ord, index
    cdef int[:] qspace_flat1 = qspace_flat
    for pix in range(numpix):
        for ord in range(order):
            index = symm[pix,ord]
            if index >= 0:
                qspace_flat1[pix] += old_q[index]

    return qspace_flat


def calc_coverage_stats_cython(qspace, qspace_radius, q_step, qlim,
                                qspace_size, num):
    cdef int i,j
    cdef int slice
    cdef int val
    cdef int overall_points = 0
    cdef int overall_covered_points = 0
    cdef int overall_redundant_points = 0

    cdef cnp.ndarray[long,ndim=1] covered_points0 = np.zeros(num, dtype=int)
    cdef cnp.ndarray[long,ndim=1] covered_points1 = np.zeros(num, dtype=int)
    cdef cnp.ndarray[long,ndim=1] covered_points2 = np.zeros(num, dtype=int)
    cdef cnp.ndarray[long,ndim=1] covered_points3 = np.zeros(num, dtype=int)
    cdef cnp.ndarray[long,ndim=1] total_points = np.zeros(num, dtype=int)

    cdef long[:] cvp0 = covered_points0
    cdef long[:] cvp1 = covered_points1
    cdef long[:] cvp2 = covered_points2
    cdef long[:] cvp3 = covered_points3
    cdef long[:] tp = total_points


    for i in range(qspace_size):
        val = qspace[i]

        if qspace_radius[i] < qlim:
            overall_points += 1

            if val > 0:
                overall_covered_points += 1

                if val > 1:
                    overall_redundant_points += 1
        slice = int(qspace_radius[i] / q_step)
        if slice < num and slice >= 0:
            tp[slice] += 1
            if val > 0:
                cvp0[slice] += 1
                if val > 1:
                    cvp1[slice] +=1
                    if val > 2:
                        cvp2[slice] += 1
                        if val > 3:
                            cvp3[slice] += 1

    return overall_points, overall_covered_points, overall_redundant_points, total_points, \
            covered_points0, covered_points1, covered_points2, covered_points3