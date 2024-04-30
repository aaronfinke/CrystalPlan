import cython
from cython import array
import array
from libc.math cimport cos, sin, M_PI, sqrt, atan2, acos
import numpy as np
cimport numpy as cnp
cnp.import_array()

cython: nonecheck=True

cdef class LimitedGoniometer:
    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cpdef float fitness_function(self, double[:] gonio_angles, double phi, double chi, double omega):

        cdef float phi_min = gonio_angles[0]
        cdef float phi_max = gonio_angles[1]
        cdef float chi_min = gonio_angles[2] 
        cdef float chi_max = gonio_angles[3] 
        cdef float omega_min = gonio_angles[4] 
        cdef float omega_max = gonio_angles[5] 
        
        cdef float phi_mid = (phi_min + phi_max) / 2
        cdef float chi_mid = (chi_min + chi_max) / 2
        cdef float omega_mid = (omega_min + omega_max) / 2

        fitness = abs(chi - chi_mid) + abs(omega - omega_mid) + abs(phi - phi_mid)

        # Big penalties for being out of the range
        if phi < phi_min: 
            fitness += (phi_min - phi) * 1.0
        if phi > phi_max:
            fitness += (phi - phi_max) * 1.0
        if chi < chi_min: 
            fitness += (chi_min - chi) * 1.0
        if chi > chi_max:
            fitness += (chi - chi_max) * 1.0
        if omega < omega_min: 
            fitness += (omega_min - omega) * 1.0
        if omega > omega_max:
            fitness += (omega - omega_max) * 1.0

        if phi < phi_min or phi > phi_max:
            fitness += 10
        if chi < chi_min or chi > chi_max:
            fitness += 10
        if omega < omega_min or omega > omega_max:
            fitness += 10
        return fitness

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cpdef cnp.ndarray[double, ndim=2] _angle_fitness_brute_cython(self, cnp.ndarray[double, ndim=1] rot_angle_list, cnp.ndarray[double, ndim=1] ending_vec,
                                    cnp.ndarray[double,ndim=2] initial_rotation_matrix, double[:] gonio_angles):
        cdef int anglenum
        cdef float rot_angle
        cdef float c, s, x, y, z
        cdef float chi, phi, omega
        cdef int i,j,k,p
        cdef cnp.ndarray[double,ndim=2] extra_rotation_matrix1 = np.empty((3,3),dtype=float)
        cdef double[:,:] extra_rotation_matrix = extra_rotation_matrix1
        cdef cnp.ndarray[double,ndim=2] total_rot_matrix1 = np.empty((3,3),dtype=float)
        cdef double[:,:] total_rot_matrix = total_rot_matrix1
        cdef double PI=3.14159265358979323846264338327950288

        cdef float ux,uy,uz 
        cdef float vx,vy,vz 
        cdef float nx,ny,nz 
        cdef float fitness, old_phi, old_chi, old_omega

        cdef int Nrot_angle_list = len(rot_angle_list)
        cdef cnp.ndarray[double, ndim=1] fitnesses1 = np.empty(Nrot_angle_list*3,dtype=float)
        cdef double[:] fitnesses = fitnesses1
        cdef cnp.ndarray[double, ndim=1] phi_list1 = np.empty(Nrot_angle_list*3,dtype=float)
        cdef double[:] phi_list = phi_list1
        cdef cnp.ndarray[double, ndim=1] chi_list1 = np.empty(Nrot_angle_list*3,dtype=float)
        cdef double[:] chi_list = chi_list1
        cdef cnp.ndarray[double, ndim=1] omega_list1 = np.empty(Nrot_angle_list*3,dtype=float)
        cdef double[:] omega_list = omega_list1

        cdef cnp.ndarray[double, ndim=2] output1 = np.empty((4,Nrot_angle_list*3),dtype=float)
        cdef double[:,:] output = output1


        for anglenum in range(Nrot_angle_list):
            rot_angle = rot_angle_list[anglenum]

            c = cos(rot_angle)
            s = sin(rot_angle)
            x = ending_vec[0]
            y = ending_vec[1]
            z = ending_vec[2]

            extra_rotation_matrix[0][0] = 1 + (1-c)*(x*x-1)
            extra_rotation_matrix[0][1] = -z*s+(1-c)*x*y
            extra_rotation_matrix[0][2] = y*s+(1-c)*x*z
            extra_rotation_matrix[1][0] = z*s+(1-c)*x*y
            extra_rotation_matrix[1][1] = 1 + (1-c)*(y*y-1)
            extra_rotation_matrix[1][2] = -x*s+(1-c)*y*z
            extra_rotation_matrix[2][0] = -y*s+(1-c)*x*z
            extra_rotation_matrix[2][1] = x*s+(1-c)*y*z
            extra_rotation_matrix[2][2] = 1 + (1-c)*(z*z-1)
            
            for i in range(3):
                for j in range(3):
                    total_rot_matrix[i][j] = 0.0
                    for k in range(3):
                        total_rot_matrix[i][j] += extra_rotation_matrix[i][k] * initial_rotation_matrix[k,j]
            
            ux = total_rot_matrix[0][0]
            uy = total_rot_matrix[1][0]
            uz = total_rot_matrix[2][0]
            vx = total_rot_matrix[0][1]
            vy = total_rot_matrix[1][1]
            vz = total_rot_matrix[2][1]
            nx = total_rot_matrix[0][2]
            ny = total_rot_matrix[1][2]
            nz = total_rot_matrix[2][2]

            if abs(vy) < 1e-8:
                chi = 0.0
                phi = atan2(nx,nz)
                omega = 0.0
            elif abs(vy+1) < 1e-8:
                chi = PI
                phi = -atan2(nx, nz)
                if phi == -PI:
                    phi = PI
                omega = 0.0
            else:
                phi = atan2(ny, uy)
                chi = acos(vy)
                omega = atan2(vz, -vx)

            old_phi = phi
            old_chi = chi
            old_omega = omega

            # try the original angles
            fitness = self.fitness_function(gonio_angles, phi, chi, omega)
            fitnesses[3*anglenum] = fitness
            phi_list[3*anglenum] = phi
            chi_list[3*anglenum] = chi
            omega_list[3*anglenum] = omega

            #make angles closer to 0
            if (phi > PI): phi -= 2*PI
            if (chi > PI): chi -= 2*PI
            if (omega > PI): omega -= 2*PI
            if (phi < -PI): phi += 2*PI
            if (chi < -PI):chi += 2*PI
            if (omega < -PI): omega += 2*PI

            fitness = self.fitness_function(gonio_angles, phi, chi, omega)
            fitnesses[3*anglenum+1] = fitness
            phi_list[3*anglenum+1] = phi
            chi_list[3*anglenum+1] = chi
            omega_list[3*anglenum+1] = omega

            # (phi-pi, -chi, omega-pi) is always equivalent
            phi = old_phi-PI
            chi = -old_chi
            omega = old_omega-PI
            if (phi > PI): phi -= 2*PI
            if (chi > PI): chi -= 2*PI
            if (omega > PI): omega -= 2*PI
            if (phi < -PI): phi += 2*PI
            if (chi < -PI): chi += 2*PI
            if (omega < -PI): omega += 2*PI
            fitness = self.fitness_function(gonio_angles, phi, chi, omega)
            fitnesses[3*anglenum+2] = fitness
            phi_list[3*anglenum+2] = phi
            chi_list[3*anglenum+2] = chi
            omega_list[3*anglenum+2] = omega


        for p in range(Nrot_angle_list*3):
            output[0][p] = fitnesses[p]
            output[1][p] = phi_list[p]
            output[2][p] = chi_list[p]
            output[3][p] = omega_list[p]

        return output1

cdef class TOPAZCryoGoniometer(LimitedGoniometer):
    cpdef float fitness_function(self, double[:] gonio_angles, double phi, double chi, double omega):
        cdef float phi_min = gonio_angles[0]
        cdef float phi_max = gonio_angles[1]
        cdef float chi_mid = gonio_angles[2]

        cdef float phi_mid
        cdef float fitness

        phi_mid = (phi_min + phi_max) / 2

        fitness = abs(chi - chi_mid)*10.0 + abs(phi - phi_mid)/10.0

        #big penalties for being out of the range

        if phi < phi_min:
            fitness += (phi_min - phi) * 1.0
        if phi > phi_max:
            fitness += (phi - phi_max) * 1.0
        return fitness

cdef class SNAPLimitedGoniometer(LimitedGoniometer):
    cpdef float fitness_function(self, double[:] gonio_angles, double phi, double chi, double omega):
        cdef float phi_min = gonio_angles[0]
        cdef float phi_max = gonio_angles[1]
        cdef float chi_mid = gonio_angles[2]

        cdef float phi_mid
        cdef float fitness

        phi_mid = (phi_min + phi_max) / 2

        fitness = abs(chi - chi_mid)*10.0 + abs(phi - phi_mid)/10.0

        #big penalties for being out of the range

        if phi < phi_min:
            fitness += (phi_min - phi) * 1.0
        if phi > phi_max:
            fitness += (phi - phi_max) * 1.0
        return fitness

cdef class MandiGoniometer(LimitedGoniometer):
    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cpdef float fitness_function(self, double[:] gonio_angles, double phi, double chi, double omega):
        """Fitness is always good since the goniometer has no limits"""
        cdef float fitness = abs(phi)
        return fitness

cdef class MandiVaryOmegaGoniometer(LimitedGoniometer):
    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cpdef float fitness_function(self, double[:] gonio_angles, double phi, double chi, double omega):
        cdef float phi_min = gonio_angles[0]
        cdef float phi_max = gonio_angles[1]
        cdef float omega_min = gonio_angles[2]
        cdef float omega_max = gonio_angles[3]
        cdef float chi_mid = gonio_angles[4]        

        cdef float phi_mid, omega_mid
        cdef float fitness

        phi_mid = (phi_min + phi_max) / 2
        omega_mid = (omega_min + omega_max) / 2

        fitness = abs(chi - chi_mid)*10.0 + abs(omega - omega_mid)/10.0 + abs(phi - phi_mid)/10.0;

        #big penalties for being out of the range

        if phi < phi_min:
            fitness += (phi_min - phi) * 1.0
        if phi > phi_max:
            fitness += (phi - phi_max) * 1.0
        if omega < omega_min:
            fitness += (omega_min - omega) * 1.0
        if omega > omega_max:
            fitness += (omega - omega_max) * 1.0
        return fitness


cdef class ImageGoniometer(LimitedGoniometer):
    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cpdef float fitness_function(self, double[:] gonio_angles, double phi, double chi, double omega):
        """Fitness is always good since the goniometer has no limits"""
        cdef float fitness = abs(phi)
        return fitness

cdef class ImageineMiniKappaGoniometer(LimitedGoniometer):
    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cpdef float fitness_function(self, double[:] gonio_angles, double phi, double chi, double omega):
        cdef float phi_min = gonio_angles[0]
        cdef float phi_max = gonio_angles[1]
        cdef float chi_min = gonio_angles[2]        
        cdef float chi_max = gonio_angles[3]        
        cdef float omega_min = gonio_angles[4]
        cdef float omega_max = gonio_angles[5]

        cdef float phi_mid, omega_mid, chi_mid
        cdef float fitness

        phi_mid = (phi_min + phi_max) / 2
        omega_mid = (omega_min + omega_max) / 2
        chi_mid = (chi_min + chi_max) / 2

        fitness =  abs(chi - chi_mid) + abs(omega - omega_mid) + abs(phi - phi_mid);
        #big penalties for being out of the range

        if phi < phi_min:
            fitness += (phi_min - phi) * 1.0
        if phi > phi_max:
            fitness += (phi - phi_max) * 1.0
        if chi < chi_min:
            fitness += (chi_min - chi) * 1.0
        if chi > chi_max:
            fitness += (chi - chi_max) * 1.0
        if omega < omega_min:
            fitness += (omega_min - omega) * 1.0
        if omega > omega_max:
            fitness += (omega - omega_max) * 1.0

        return fitness


cdef class TopazAmbientGoniometer(LimitedGoniometer):
    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cpdef float fitness_function(self, double[:] gonio_angles, double phi, double chi, double omega):
        cdef float phi_min = gonio_angles[0]
        cdef float phi_max = gonio_angles[1]
        cdef float omega_min = gonio_angles[2]
        cdef float omega_max = gonio_angles[3]
        cdef float chi_mid = gonio_angles[4]        

        cdef float phi_mid, omega_mid
        cdef float fitness

        phi_mid = (phi_min + phi_max) / 2
        omega_mid = (omega_min + omega_max) / 2

        fitness = abs(chi - chi_mid)*10.0 + abs(omega - omega_mid)/10.0 + abs(phi - phi_mid)/10.0;

        #big penalties for being out of the range

        if phi < phi_min:
            fitness += (phi_min - phi) * 1.0
        if phi > phi_max:
            fitness += (phi - phi_max) * 1.0
        if omega < omega_min:
            fitness += (omega_min - omega) * 1.0
        if omega > omega_max:
            fitness += (omega - omega_max) * 1.0
        return fitness
            
cdef class HB3AGoniometer(LimitedGoniometer):
    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cpdef float fitness_function(self, double[:] gonio_angles, double phi, double chi, double omega):
        cdef double center = 3.14159*25.0/180.0
        cdef double omegadiff = omega - center
        cdef float result
        if omegadiff < 0:
            omegadiff = -omegadiff
        result = abs(chi) + omegadiff + abs(phi)/1000.0
        return result


cdef class CorelliGoniometer(LimitedGoniometer):
    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cpdef float fitness_function(self, double[:] gonio_angles, double phi, double chi, double omega):
        cdef float phi_min = gonio_angles[0]
        cdef float phi_max = gonio_angles[1]
        cdef float omega_min = gonio_angles[2]
        cdef float omega_max = gonio_angles[3]
        cdef float chi_mid = gonio_angles[4]        

        cdef float phi_mid, omega_mid
        cdef float fitness

        phi_mid = (phi_min + phi_max) / 2
        omega_mid = (omega_min + omega_max) / 2

        fitness = abs(chi - chi_mid)*10.0 + abs(omega - omega_mid)/10.0 + abs(phi - phi_mid)/10.0;

        #big penalties for being out of the range

        if phi < phi_min:
            fitness += (phi_min - phi) * 1.0
        if phi > phi_max:
            fitness += (phi - phi_max) * 1.0
        if omega < omega_min:
            fitness += (omega_min - omega) * 1.0
        if omega > omega_max:
            fitness += (omega - omega_max) * 1.0
        return fitness

