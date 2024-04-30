import cython
from libc.math cimport cos, sin, M_PI, sqrt
import numpy as np
cimport numpy as cnp
cnp.import_array()

ctypedef fused double_long:
    double
    long
@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef cnp.ndarray[int, ndim=4] calculate_coverage_cython(list xlist, list ylist, double[:,:] azimuthal_angle, double[:,:] elevation_angle,
                              double[:,:] rot_matrix, long set_value1, long set_value2, long number_of_ints, long[:,:,:,:] coverage,
                              double stride, long max_index, double wl_min, double wl_max, double qlim, double q_resolution):
    cdef double az, elev, q_length
    cdef long numfrac, index
    cdef long i
    cdef int ix, iy, iix, iiy
    cdef double lim_min, lim_max
    cdef double q_min_x, q_min_y, q_min_z
    cdef double dx, dy, dz
    cdef double qx, qy, qz
    cdef long iqx, iqy, iqz
    cdef (double, double, double) q_min, q_max
    cdef int xlen = len(xlist)
    cdef int ylen = len(ylist)

    for iix in range(xlen):
        ix = xlist[iix]

        for iiy in range(ylen):
            iy = ylist[iiy]
            # Angles of the detector pixel positions
            az = azimuthal_angle[iy,ix]
            elev = elevation_angle[iy,ix]
            q_min = getq_cython(wl_min, wl_min, az, elev, rot_matrix)
            q_max = getq_cython(wl_max, wl_max, az, elev, rot_matrix)
            shrink_q_vector(q_min, qlim)
            shrink_q_vector(q_max, qlim)

            q_length = (vector_length(q_max) - vector_length(q_min))

            if q_length < 0:
                q_length = -q_length

            # How many steps will we take. The multiplication factor here is a fudge to make sure it covers fully.
            numfrac = int((1.25 * q_length) / (q_resolution))

            if numfrac > 0:
                # There is something to measure
                dx = (float(q_max[0]) - float(q_min[0])) / numfrac
                dy = (float(q_max[1]) - float(q_min[1])) / numfrac
                dz = (float(q_max[2]) - float(q_min[2])) / numfrac

                lim_min = -qlim
                lim_max = +qlim

                q_min_x = float(q_min[0])
                q_min_y = float(q_min[1])
                q_min_z = float(q_min[2])

                for i in range(numfrac):
                    qx = q_min_x + i * dx
                    iqx = round((qx - lim_min) / q_resolution)
                    if (iqx >= 0) and (iqx < stride):
                        qy = q_min_y + i * dy
                        iqy = round((qy - lim_min) / q_resolution)
                        if (iqy >= 0) and (iqy < stride):
                            qz = q_min_z + i * dz
                            iqz = round((qz - lim_min) / q_resolution)
                            if (iqz >= 0) and (iqz < stride):
                                if number_of_ints == 2:
                                    coverage[iqx,iqy,iqz,0] |= set_value1
                                    coverage[iqx,iqy,iqz,1] |= set_value2
                                else:
                                    coverage[iqx,iqy,iqz,0] |= set_value1
    return np.asarray(coverage)


cdef double vector_length((double, double, double) vector):
    cdef double length = 0.0
    for i in range(len(vector)):
        length += float(vector[i]) * float(vector[i])
    return sqrt(length)

cdef (double, double, double) shrink_q_vector ((double, double, double) q, double limit):
    cdef double length = vector_length(q)
    cdef list q1 = [0.0,0.0,0.0]
    cdef int i
    if length <= 0:
        return q
    for i in range(len(q)):
        if length > limit:
            q1[i] = q[i] * (limit/length)
        else:
            q1[i] = q[i]
    return tuple(q1)


cpdef (double, double, double) getq_cython(double wl_input, double wl_output, double az, double elevation, double[:,:] rot_matrix):
    # The scattered beam emanates from the centre of this spher.
    # Find the intersection of the scattered beam and the sphere, in XYZ
    # We start with an Ewald sphere of radius 1/wavelength
    cdef double pi = M_PI
    cdef double qx, qy, qz
    cdef double q0, q1, q2
    # Assuming azimuth of zero points to z positive = same direction as incident radiation.
    cdef double r2, x, y, z, incident_z
    r2 = cos(elevation) / wl_output
    z = cos(az) * r2
    x = sin(az) * r2

    # Assuming elevation angle is 0 when horizontal, positive to y positive:
    y = sin(elevation) / wl_output

    # And here is the incident beam direction: Along the z-axis
    incident_z = 1.0 / wl_input

    # The vector difference between the two is the q vector
    qx = 2 * pi * x
    qy = 2 * pi * y
    qz = 2 * pi * (z - incident_z)

    # Now we switch to the coordinate system of the crystal.
    # The scattered beam direction (the detector location) is rotated relative to the crystal because the sample is rotated.
    # So is the incident beam direction.
    # Therefore, the q-vector measured is simply rotated
    q0 = qx * rot_matrix[0,0] + qy * rot_matrix[0,1] + qz * rot_matrix[0,2]
    q1 = qx * rot_matrix[1,0] + qy * rot_matrix[1,1] + qz * rot_matrix[1,2]
    q2 = qx * rot_matrix[2,0] + qy * rot_matrix[2,1] + qz * rot_matrix[2,2]

    return q0, q1, q2

cpdef (double, double, double, double, double, double) getq_inelastic_cython(double wl_input, double wl_output, double az, double elevation, double[:,:] rot_matrix):
    # The scattered beam emanates from the centre of this spher.
    # Find the intersection of the scattered beam and the sphere, in XYZ
    # We start with an Ewald sphere of radius 1/wavelength
    cdef double pi = M_PI
    cdef double qx, qy, qz
    cdef double q0, q1, q2, q3, q4, q5
    # Assuming azimuth of zero points to z positive = same direction as incident radiation.
    cdef double r2, x, y, z, incident_z
    r2 = cos(elevation) / wl_output
    z = cos(az) * r2
    x = sin(az) * r2

    # Assuming elevation angle is 0 when horizontal, positive to y positive:
    y = sin(elevation) / wl_output

    # And here is the incident beam direction: Along the z-axis
    incident_z = 1.0 / wl_input

    # The vector difference between the two is the q vector
    qx = 2 * pi * x
    qy = 2 * pi * y
    qz = 2 * pi * (z - incident_z)

    # Now we switch to the coordinate system of the crystal.
    # The scattered beam direction (the detector location) is rotated relative to the crystal because the sample is rotated.
    # So is the incident beam direction.
    # Therefore, the q-vector measured is simply rotated
    q0 = qx * rot_matrix[0,0] + qy * rot_matrix[0,1] + qz * rot_matrix[0,2]
    q1 = qx * rot_matrix[1,0] + qy * rot_matrix[1,1] + qz * rot_matrix[1,2]
    q2 = qx * rot_matrix[2,0] + qy * rot_matrix[2,1] + qz * rot_matrix[2,2]
    q3 = qx
    q4 = qy
    q5 = qz
    return q0, q1, q2, q3, q4, q5

cpdef cnp.ndarray[short,ndim=3] get_coverage_cython(int number_of_ints, cnp.ndarray[short,ndim=3] coverage, int mask1, int mask2, list coverage_list):
    cdef short[:,:,:] coverages = coverage
    for one_coverage in coverage_list:
        #By applying the mask and the >0 we take away any unwanted detectors.
        if number_of_ints == 1:
            #coverage = coverage + ((one_coverage & mask) != 0)
            coverages += ((one_coverage[:, :, :, 0] & mask1) != 0)
        else:
            coverages += (((one_coverage[:, :, :, 0] & mask1) | (one_coverage[:, :, :, 1] & mask2)) != 0)
    return coverage

@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef cnp.ndarray[double, ndim=3] calculate_coverage_inelastic_cython(double wl_input, list xlist, list ylist, double[:,:] azimuthal_angle, 
                                                                   double[:,:] elevation_angle, double[:,:] rot_matrix, double energy_constant,
                                                                   double ki_squared, double ki, double[:,:,:] coverage, long stride, long max_index,
                                                                   double wl_min, double wl_max, double qlim, double q_resolution):
    cdef double kfz, kf_squared, E
    cdef double lim_min = -qlim,
    cdef double lim_max = +qlim
    cdef double az, elev
    cdef int i, ix, iy
    cdef (double, double, double, double, double, double) q_max_both, q_min_both
    cdef double[3] q_max, q_min, q_max_unrot, q_min_unrot
    cdef double q_max_length
    cdef double q_min_length
    cdef double[3] q_diff, q_diff_unrot
    cdef double q_length
    cdef long numfrac
    cdef double dx, dy, dz
    cdef double qx, qy, qz
    cdef long index
    cdef double qx_unrot, qy_unrot, qz_unrot
    cdef long iqx, iqy, iqz

    for ix in xlist:
        for iy in ylist:
            az = azimuthal_angle[iy,ix]
            elev = elevation_angle[iy,ix]
            q_min_both = getq_inelastic_cython(wl_input, wl_min, az, elev, rot_matrix)
            q_max_both = getq_inelastic_cython(wl_input, wl_max, az, elev, rot_matrix)

            q_max_length = 0
            q_min_length = 0
            for i in range(3):
                q_max[i] = q_max_both[i]
                q_max_unrot[i] = q_max_both[i+3]
                q_max_length += q_max[i]*q_max[i]

                q_min[i] = q_min_both[i]
                q_min_unrot[i] = q_min_both[i+3]
                q_min_length += q_min[i]*q_min[i]
            q_max_length = sqrt(q_max_length)
            q_min_length = sqrt(q_min_length)

            q_length = 0.0
            for i in range(3):
                q_diff[i] = q_max[i] - q_min[i]
                q_diff_unrot[i] = q_max_unrot[i] - q_min_unrot[i]
                q_length += q_diff[i]*q_diff[i]
            q_length = sqrt(q_length)
            
            numfrac = int((1.25*q_length)/(q_resolution))

            if numfrac > 0:
                dx = q_diff[0] / numfrac
                dy = q_diff[1] / numfrac
                dz = q_diff[2] / numfrac

                dx_unrot = q_diff_unrot[0] / numfrac
                dy_unrot = q_diff_unrot[1] / numfrac
                dz_unrot = q_diff_unrot[2] / numfrac

                for i in range(numfrac):
                    qx = q_min[0] + i*dx
                    qx_unrot = q_min_unrot[0] + i*dx_unrot
                    iqx = int(round((qx - lim_min) / q_resolution))
                    if iqx >= 0 and iqx < stride:
                        qy = q_min[1] + i*dy
                        qy_unrot = q_min_unrot[1] + i*dy_unrot
                        iqy = int(round((qy - lim_min) / q_resolution))
                        if iqy >= 0 and iqy < stride:
                            qz = q_min[2] + i*dz
                            qz_unrot = q_min_unrot[2] + i*dz_unrot
                            iqz = long(round((qz - lim_min) / q_resolution))
                            if iqz >= 0 and iqz < stride:
                                kfz = ki + qz_unrot

                                kf_squared = qx_unrot*qx_unrot + qy_unrot*qy_unrot + kfz*kfz
                                E = energy_constant * (kf_squared - ki_squared)
                                coverage[iqx,iqy,iqz] = E
    
    return np.asarray(coverage)












