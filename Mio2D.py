import numpy as np
from numba import jit, njit, prange



@njit
def f (x, y, R_1, R_2):
    """internal calculation of the kick function
    Parameters
    ----------
    x : float
        position
    y : float
        position
    R_1 : float
        inner lens radius
    R_2 : float
        outer lens radius
    Returns
    -------
    float 
        kick value
    """
    r=np.sqrt(x**2 + y**2)
    if r < R_1:
        return 0.0
        
    if r > R_2:
        return 1.0

    else :
        result= (r**2 - R_1**2)/(R_2**2 - R_1**2)
        return result


def make_correlated_noise(n_elements, gamma=0.0):
    """Make an array of correlated noise
    
    Parameters
    ----------
    n_elements : unsigned int
        number of elements
    gamma : float, optional
        correlation coefficient, by default 0.0
    
    Returns
    -------
    ndarray
        the noise array
    """
#    correl=5
 #   noise=np.empty(0)
  #  for s in range(int(n_elements/correl)):
   #     turn=np.full(correl, np.random.binomial(1, 0.5, 1))
    #    noise=np.append(noise, turn)  
    #noise=band_limited_noise(0.305, 0.31, n_elements)
    #print(np.mean(noise)) 
    #print(noise)
    #np.random.seed(1)
    #noise=np.full(n_elements, 0.5)
    noise=np.zeros(n_elements)
    #noise=np.random.rand(n_elements)
    # noise = np.random.normal(0.0, 1.0, n_elements)
    #noise = np.random.binomial(1, 0.5, n_elements)
    
    #print(len(noise))
    if gamma != 0.0:
        for i in range(1, n_elements):
            noise[i] += gamma * noise[i - 1]
    return noise




@njit
def iterate(x, px, y, py, noise, epsilon, omega_0x, omega_1x, omega_2x, omega_0y, omega_1y, omega_2y, omega_1xy, omega_2xy, omega_2yx, R_1, R_2, TH_MAX, barrier_radius, start):
    """internal iteration method for symplectic map
    
    Parameters
    ----------
    x : float
        x0
    px : float
        px0
    y : float
        y0
    py : float
        py0
    noise : ndarray
        array of noise values
    epsilon : float
        epsilon value
    
    omega_vari : float
        all the omega values 
     R_1 : float
        inner lens radius
    R_2 : float
        outer lens radius
    TH_MAX : float
        theta max value
    barrier_radius : float
        barrier position 
    start : unsigned int
        starting iteration value
    
    Returns
    -------
    (float, float, unsigned int)
        (x, px, y, py, iterations)
    """    

    
    for i in range(len(noise)):
        g=f(x, y, R_1, R_2)
        
        
        if (x==0) and (px==0):
            return 0.0, 0.0, 0.0, 0.0, start
        
        temp1x = x
        temp2x = (px + (epsilon * 
         noise[i] * TH_MAX * g * R_2 * 914099.8357243269 * x/ (x**2 + y**2) ))
        temp1y = y
        temp2y = (py + (epsilon * 
         noise[i] * TH_MAX * g * R_2 * 914099.8357243269 * y/ (x**2 + y**2) ))

        action_y = (temp1y * temp1y + temp2y * temp2y) * 0.5
        action_x = (temp1x * temp1x + temp2x * temp2x) * 0.5
        if ((action_x + action_y) >= (barrier_radius**2 * 0.5)):
            return 0.0, 0.0, 0.0, 0.0, i + start 

        rot_angle_x = omega_0x + (omega_1x * action_x) + (omega_1xy * action_y) + (0.5*  omega_2x * action_x * action_x)+ ( omega_2yx * action_x * action_y)+ (0.5*  omega_2xy * action_y * action_y)
        rot_angle_y = omega_0y + (omega_1y * action_y) + (omega_1xy * action_x) + (0.5*  omega_2y * action_y * action_y)+ ( omega_2xy * action_x * action_y)+ (0.5*  omega_2yx * action_x * action_x)
        
        x = np.cos(rot_angle_x) * temp1x + np.sin(rot_angle_x) * temp2x
        px = -np.sin(rot_angle_x) * temp1x + np.cos(rot_angle_x) * temp2x
        
        y = np.cos(rot_angle_y) * temp1y + np.sin(rot_angle_y) * temp2y
        py = -np.sin(rot_angle_y) * temp1y + np.cos(rot_angle_y) * temp2y
        
        action_y = (y * y + py * py) * 0.5
        action_x = (x * x + px * px) * 0.5
        if ((action_x + action_y) >= (barrier_radius**2 * 0.5)):
        #if (np.sqrt(x**2 + y**2) >= barrier_radius):   
            return 0.0, 0.0, 0.0, 0.0, i + start 
        

    return x, px, y, py, i + start +1

@njit(parallel=True)
def symplectic_map_personal(x, px, y, py, step_values, n_iterations, epsilon, omega_0x, omega_1x, omega_2x, omega_0y, omega_1y, omega_2y, omega_1xy, omega_2xy, omega_2yx, R_1, R_2, TH_MAX, barrier_radius, gamma=0.0):
    """computation for personal noise symplectic map
    
    Parameters
    ----------
    x : ndarray
        x initial condition
    px : ndarray
        px initial condition
    step_values : ndarray
        iterations already performed
    n_iterations : unsigned int
        number of iterations to perform
    epsilon : float
        epsilon value
    omega_vari : float
        all the omega values

    R_1 : float
        inner radius
    R_2 : float
        outer radius
    TH_MAX : fload
        theta max value
    barrier_radius : float
        barrier radius
    
    gamma : float, optional
        correlation coefficient, by default 0.0
    
    Returns
    -------
    (ndarray, ndarray, ndarray)
        x, px, step_values
    """    
    for i in prange(len(x)):
        personal_noise = make_correlated_noise(n_iterations, gamma)
        x[i], px[i], y[i], py[i], step_values[i] = iterate(x[i], px[i], y[i], py[i], personal_noise, epsilon, omega_0x, omega_1x, omega_2x, omega_0y, omega_1y, omega_2y, omega_1xy, omega_2xy, omega_2yx, R_1, R_2, TH_MAX, barrier_radius, step_values[i])
    return x, px, y, py, step_values 



@njit(parallel=True)
def symplectic_map_common(x, px, y, py, step_values, noise_array, epsilon, omega_0x, omega_1x, omega_2x, omega_0y, omega_1y, omega_2y, omega_1xy, omega_2xy, omega_2yx, R_1, R_2, TH_MAX, barrier_radius, gamma=0.0):
    """computation for personal noise symplectic map

    Parameters
    ----------
    x : ndarray
        x initial condition
    px : ndarray
        px initial condition
    step_values : ndarray
        iterations already performed
    noise_array : ndarray
        noise array for the whole group
    epsilon : float
        epsilon value
    omega_vari : float
        all the omega values
    
    R_1 : float
        inner radius
    R_2 : float
        outer radius
    TH_MAX : fload
        theta max value
    barrier_radius : float
        barrier radius
    gamma : float, optional
        correlation coefficient, by default 0.0

    Returns
    -------
    (ndarray, ndarray, ndarray)
        x, px, step_values
    """
    for i in prange(len(x)):
        x[i], px[i], y[i], py[i], step_values[i] = iterate(x[i], px[i], y[i], py[i], noise_array, epsilon, omega_0x, omega_1x, omega_2x, omega_0y, omega_1y, omega_2y, omega_1xy, omega_2xy, omega_2yx, R_1, R_2, TH_MAX, barrier_radius,  step_values[i])
    return x, px, y, py, step_values 