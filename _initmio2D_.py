import numpy as np
from tqdm import tqdm
import scipy as sc
import scipy.integrate as integrate
from numba import cuda
import numba.cuda.random as random
import Mio2D as mio




def f(x, y, R_1, R_2):
    return mio.f(x, y, R_1, R_2)


def make_correlated_noise(n_elements, gamma=0.0):
    return mio.make_correlated_noise(n_elements, gamma)


class symplectic_map(object):
    def __init__(self):
        pass
    
    def reset(self):
        pass

    def common_noise(self):
        pass
    
    def personal_noise(self):
        pass
    
    def get_data(self):
        """Get data from engine.
        
        Returns
        -------
        tuple(ndarray, ndarray, ndarray)
            tuple with x, p, and number of iterations before loss data
        """
        return self.x, self.px, self.times

    def get_data_y(self):
        """Get data from engine.
        
        Returns
        -------
        tuple(ndarray, ndarray, ndarray)
            tuple with x, p, and number of iterations before loss data
        """
        return self.y, self.py, self.times

    def get_filtered_data(self):
        """Get filtered data from engine.
        
        Returns
        -------
        tuple(ndarray, ndarray, ndarray)
            tuple with x, p, and number of iterations before loss data
        """
        t = self.times
        return (self.x)[t >= self.iterations], (self.px)[t >= self.iterations], t[t >= self.iterations]

    def get_filtered_data_y(self):
        """Get filtered data from engine.
        
        Returns
        -------
        tuple(ndarray, ndarray, ndarray)
            tuple with x, p, and number of iterations before loss data
        """
        t = self.times
        return (self.y)[t >= self.iterations], (self.py)[t >= self.iterations], t[t >= self.iterations]
   
    def get_th(self):
        """Find theta given x and p.
        
        Returns
        -------
        ndarray
            angle array data
        """
        x, p, t = self.get_filtered_data()
        th=[]
        for i in range(len(x)):
            th_1=np.arcsin(x[i]/np.sqrt((x[i] * x[i])+(p[i] * p[i])))
            th_2=np.arccos(p[i]/np.sqrt((x[i] * x[i])+(p[i] * p[i])))
            if np.sin(th_1) > 0 and np.cos(th_2) > 0 :
                th.append(th_1)
            if np.sin(th_1) > 0 and np.cos(th_2) < 0 :
                th.append(th_2)
            if np.sin(th_1) < 0 and np.cos(th_2) > 0 :
                th_4=(th_1 + (np.pi*2))
                th.append(th_4)
            if np.sin(th_1) < 0 and np.cos(th_2) < 0 :
             th_3=(np.pi - th_1)
             th.append(th_3)
            
        Th=np.array(th)
        return Th

    def get_action(self):
        """Get action data from engine
        
        Returns
        -------
        ndarray
            action array data
        """
        return (self.x * self.x + self.px * self.px) * 0.5

    def get_filtered_action(self):
        """Get filtered action data from engine (i.e. no zero values of lost particles)
        
        Returns
        -------
        ndarray
            filtered action array data
        """
        action = self.get_action()
        return action[action > 0]

    def get_times(self):
        """Get times from engine
        
        Returns
        -------
        ndarray
            times array
        """
        return self.times

    def get_filtered_times(self):
        """Get only loss times from engine (i.e. only loss particles)
        
        Returns
        -------
        ndarray
            filtered times array
        """
        times = self.get_times()
        return times[times < self.iterations]

    def get_survival_quota(self):
        """Get time evolution of number of survived particles
        
        Returns
        -------
        ndarray
            time evolution of survived particles
        """
        t = np.array(self.get_times())
        max_t = int(np.amax(t))
        quota = np.empty(max_t)
        for i in range(max_t):
            quota[i] = np.count_nonzero(t>i)
        return quota

    def get_lost_particles(self):
        """Get time evolution of lost particles
        
        Returns
        -------
        ndarray
            time evolution of number of lost particles
        """
        quota = self.get_survival_quota()
        return self.N - quota

    def current_binning(self, bin_size):
        """Execute current binning and computation
        
        Parameters
        ----------
        bin_size : int
            size of the binning to consider for current computation
        
        Returns
        -------
        tuple(ndarray, ndarray)
            array with corresponding sampling time (middle point), current value computed.
        """
        survival_quota = self.get_survival_quota()
        
        points = [i for i in range(0, len(survival_quota), bin_size)]
        if len(survival_quota) % bin_size == 0:
            points.append(len(survival_quota) - 1)
        t_middle = [(points[i + 1] + points[i]) *
                    0.5 for i in range(len(points) - 1)]
        currents = [(survival_quota[points[i]] - survival_quota[points[i+1]]
                     ) / bin_size for i in range(len(points) - 1)]
        return np.array(t_middle), np.array(currents)

    @staticmethod
    def generate_instance(omega_0x, omega_1x, omega_2x, omega_0y, omega_1y, omega_2y, omega_1xy, omega_2xy, omega_2yx, epsilon, R_1, R_2, TH_MAX, barrier_radius, x_0, px_0, y_0, py_0, cuda_device=None):
        
        
        return symplectic_map_cpu(omega_0x, omega_1x, omega_2x, omega_0y, omega_1y, omega_2y, omega_1xy, omega_2xy, omega_2yx, epsilon, R_1, R_2, TH_MAX,  barrier_radius, x_0, px_0, y_0, py_0)

class symplectic_map_cpu(symplectic_map):
    def __init__(self, omega_0x, omega_1x, omega_2x, omega_0y, omega_1y, omega_2y, omega_1xy, omega_2xy, omega_2yx, epsilon, R_1, R_2, TH_MAX, barrier_radius, x_0, px_0, y_0, py_0):
        """Init symplectic map object!
        
        Parameters
        ----------
        object : self
            self
        omega_0 : float
            Omega 0 frequency
        omega_1 : float
            Omega 1 frequency
        omega_2 : float
            Omega 2 frequency
        epsilon : float
            Noise coefficient
        
        barrier_radius : float
            barrier position (x coordinates!)
        x_0 : ndarray
            1D array of x initial positions
        p_0 : ndarray
            1D array of p initial values
        """
        self.omega_0x = omega_0x
        self.omega_1x = omega_1x
        self.omega_2x = omega_2x
        self.omega_0y = omega_0y
        self.omega_1y = omega_1y
        self.omega_2y = omega_2y
        self.omega_1xy = omega_1xy
        self.omega_2xy = omega_2xy
        self.omega_2yx = omega_2yx
        self.epsilon = epsilon
        self.R_1 = R_1
        self.R_2 = R_2
        self.TH_MAX = TH_MAX
        
        self.barrier_radius = barrier_radius
        self.action_radius = barrier_radius ** 2 * 0.5
        self.x_0 = x_0.copy()
        self.px_0 = px_0.copy()
        self.y_0 = y_0.copy()
        self.py_0 = py_0.copy()
        self.N = len(x_0)

        self.iterations = 0

        self.x = x_0.copy()
        self.px = px_0.copy()
        self.y = y_0.copy()
        self.py = py_0.copy()
        self.times = np.zeros(len(self.x))

    def reset(self):
        """Reset the engine to initial conditions
        """
        self.iterations = 0
        self.x = self.x_0.copy()
        self.px = self.px_0.copy()
        self.y = self.y_0.copy()
        self.py = self.py_0.copy()
        self.times = np.zeros(len(self.x))

    def common_noise(self, noise_array):
        """Execute iterations with given noise array, common for all particles.
        
        Parameters
        ----------
        noise_array : ndarray
            noise array to use for computation
        """
        self.iterations += len(noise_array)
        self.x, self.px, self.y, self.py, self.times = mio.symplectic_map_common(
            self.x, self.px, self.y, self.py, self.times, noise_array, self.epsilon, self.omega_0x, self.omega_1x, self.omega_2x, self.omega_0y, self.omega_1y, self.omega_2y, self.omega_1xy, self.omega_2xy, self.omega_2yx, self.R_1, self.R_2, self.TH_MAX, self.barrier_radius
        )

    def personal_noise(self, n_iterations, gamma=0.0):
        """Execute iterations with correlated noise with different realization for every single particle.
        
        Parameters
        ----------
        n_iterations : unsigned int
            number of iterations to perform
        gamma : float, optional
            correlation coefficient (between 0 and 1!), by default 0.0
        """
        self.iterations += n_iterations
        self.x, self.px, self.y, self.py, self.times = mio.symplectic_map_personal(
            self.x, self.px, self.y, self.py, self.times, n_iterations, self.epsilon, self.omega_0x, self.omega_1x, self.omega_2x, self.omega_0y, self.omega_1y, self.omega_2y, self.omega_1xy, self.omega_2xy, self.omega_2yx, self.R_1, self.R_2, self.TH_MAX, self.barrier_radius, gamma
        )


