import cython
from libc.math cimport cos, sin, M_PI, sqrt
import numpy as np
cimport numpy as cnp
ctypedef fused double_long:
    double
    long
@cython.cdivision(True)
cdef double[:,:,:,:] calculate_coverage_cython(int[:] xlist, int[:] ylist, double[:,:] azimuthal_angle, double[:,:] elevation_angle,
                              double[:] rot_matrix, long set_value1, long set_value2, long number_of_ints, cnp.ndarray[int, ndim=4] coverage,
                              double stride, long max_index, double wl_min, double wl_max, double qlim, double q_resolution):
    cdef long[:,:,:,:] coverages = coverage
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

    for iix in range(len(xlist)):
        ix = xlist[iix]

        for iiy in range(len(ylist)):
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
            numfrac = int(1.25 * q_length) / (q_resolution)

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
                                    coverages[iqx,iqy,iqz,0] |= set_value1
                                    coverages[iqx,iqy,iqz,1] |= set_value2
                                else:
                                    coverages[iqx,iqy,iqz,0] |= set_value1
    return coverage


cdef double vector_length((double, double, double) vector):
    cdef double length = 0.0
    for i in range(len(vector)):
        length += float(vector[i]) * float(vector[i])
    return sqrt(length)

cdef void shrink_q_vector ((double, double, double) q, double limit):
    cdef double length = vector_length(q)
    if length <= 0:
        return
    for i in range(len(q)):
        if length > limit:
            q[i] = float(q[i]) * (limit/length)
        else:
            q[i] = float(q[i])


cdef (double, double, double) getq_cython(double wl_input, double wl_output, double az, double elevation, double[:] rot_matrix):
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
    q0 = qx * rot_matrix[0+0] + qy * rot_matrix[0+1] + qz * rot_matrix[0+2]
    q1 = qx * rot_matrix[3+0] + qy * rot_matrix[3+1] + qz * rot_matrix[3+2]
    q2 = qx * rot_matrix[6+0] + qy * rot_matrix[6+1] + qz * rot_matrix[6+2]

    return q0, q1, q2

cdef (double, double, double, double, double, double) getq_inelastic_cython(double wl_input, double wl_output, double az, double elevation, double* rot_matrix):
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
    q0 = qx * rot_matrix[0+0] + qy * rot_matrix[0+1] + qz * rot_matrix[0+2]
    q1 = qx * rot_matrix[3+0] + qy * rot_matrix[3+1] + qz * rot_matrix[3+2]
    q2 = qx * rot_matrix[6+0] + qy * rot_matrix[6+1] + qz * rot_matrix[6+2]
    q3 = qx
    q4 = qy
    q5 = qz
    return q0, q1, q2, q3, q4, q5

