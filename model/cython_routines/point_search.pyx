import cython
from libc.math cimport fabs, sqrt
import numpy as np

"""
'base_point', 'horizontal', 'vertical', 'normal'
'beam', 'array_size', 'n_dot_base', 'height', 'width', 'wl_min', 'wl_max'
"""
@cython.cdivision(True)
def point_search(double[:,:] base_point, double[:,:] horizontal, double[:,:] vertical, double[:,:] normal,
                 double[:,:] beam, int array_size, int n_dot_base, int height, int width, double wl_min, double wl_max):
    cdef double az, elev
    cdef double b_x, b_y, b_z
    cdef double x, y, z, temp
    cdef double h, v, d
    cdef double diff_x, diff_y, diff_z
    cdef double projection, beam_length,  wavelength

    h_out_m = np.empty(array_size)
    v_out_m = np.empty(array_size)
    wl_out_m = np.empty(array_size)
    distance_out_m = np.empty(array_size)
    hits_it_m = np.empty(array_size, dtype='int')

    cdef double[:] h_out = h_out_m
    cdef double[:] v_out = v_out_m
    cdef double[:] wl_out = wl_out_m
    cdef double[:] distance_out = distance_out_m
    cdef long[:] hits_it = hits_it_m

    base_point_x = base_point[0][0]
    base_point_y = base_point[1][0]
    base_point_z = base_point[2][0]
    horizontal_x = horizontal[0][0]
    horizontal_y = horizontal[1][0]
    horizontal_z = horizontal[2][0]
    vertical_x = vertical[0][0]
    vertical_y = vertical[1][0]
    vertical_z = vertical[2][0]
    nx = normal[0][0]
    ny = normal[1][0]
    nz = normal[2][0]
    n_dot_base_f = float(n_dot_base)

    cdef int i
    cdef int error_count = 0
    cdef int bad_beam = 0

    cdef double min = 1e-6

    for i in range(array_size):
        bad_beam = 0
        # non-normalized beam direction
        b_x = beam[0,i]
        b_y = beam[1,i]
        b_z = beam[2,i]
        # so we normalize it
        beam_length = sqrt(b_x*b_x + b_y*b_y + b_z*b_z)
        try:
            b_x = b_x/beam_length
        except ZeroDivisionError:
            b_x = float("nan")
        try:
            b_y = b_y/beam_length
        except ZeroDivisionError:
            b_y = float("nan")
        try:
            b_z = b_z/beam_length
        except ZeroDivisionError:
            b_z = float("nan")
        # Check if the wavelength is within range
        try:
            wavelength = 6.2831853071795862/beam_length
        except ZeroDivisionError:
            wavelength = float("nan")
        if (wavelength <= wl_max) and (wavelength >= wl_min):
            # Wavelength is in range! Keep going.

            # Make sure the beam points in the same direction as the detector, not opposite to it
            # project beam onto detector's base_point
            projection = (base_point_x*b_x)+(base_point_y*b_y)+(base_point_z*b_z)
            if projection > 0:
                # beam points toward the detector

                # This beam coincides with the origin (0,0,0)
                # Therefore the line equation is x/bx = y/by = z/bz

                # Now we look for the intersection between the plane of normal nx,ny,nz and the given angle.
                # threshold to avoid dividebyzero
                if fabs(b_z) > min:
                    z = n_dot_base_f / ((nx * b_x) / b_z + (ny *b_y) / b_z + nz)
                    temp = z/b_z
                    y = b_y * temp
                    x = b_x * temp
                elif fabs(b_y) > min:
                    y = n_dot_base_f / (nx*b_x/b_y + ny + nz*b_z/b_y)
                    temp = (y / b_y)
                    x = b_x * temp
                    z = b_z * temp
                elif fabs(b_x) > min:
                    x = n_dot_base_f / (nx + ny *b_y / b_x + nz * b_z / b_x)
                    temp = (x / b_x)
                    y = b_y * temp
                    z = b_z * temp
                else:
                    # the scattered beam is 0,0,0
                    error_count += 1
                    bad_beam = 1
            else:
                # The projection is < 0
                # means the projection is opposite the detector
                bad_beam = 1
        else:
            # beam is opposite detector
            bad_beam = 1

        if bad_beam:
            h_out[i] = float("nan")
            v_out[i] = float("nan")
            wl_out[i] = wavelength
            hits_it[i] = 0
        else:
            # valid beam calculation
            # difference between this point and the base point (the cneter)
            diff_x = x - base_point_x
            diff_y = y - base_point_y
            diff_z = z - base_point_z

            # Project onto horizontal and vertical axes by doing a dot product
            h = diff_x * horizontal_x + diff_y * horizontal_y + diff_z * horizontal_z
            v = diff_x * vertical_x + diff_y * vertical_y + diff_z * vertical_z

            # save to matrix
            h_out[i] = h
            v_out[i] = v
            wl_out[i] = wavelength

            distance_out[i] = sqrt(x*x + y*y + z*z)
            if (v > -height/2) and (v < height/2) and (h > -width/2) and (h < width/2):
                hits_it[i] = 1
            else:
                hits_it[i] = 0

    return_val = error_count

    return return_val,h_out_m, v_out_m, wl_out_m, distance_out_m, hits_it_m






