from numpy.core.arrayprint import _guarded_repr_or_str
from numpy.core.function_base import linspace
from _initmio2D_ import make_correlated_noise, f, symplectic_map as sm
import numpy as np
import matplotlib.pyplot as plt
import  crank_nicolson_numba.generic as cn
import scipy.integrate as integrate
from numba import njit



#@njit()
def interpolation(data, index):
    #if np.any(np.isnan(data)):
     #   value = np.nan
    if index == 0:
        value = 1
    else:
        if index == len(data) - 1:
            index -= 1
        cf1 = np.absolute(data[index - 1])
        cf2 = np.absolute(data[index])
        cf3 = np.absolute(data[index + 1])
       
        if cf3 > cf1:
            p1 = cf2
            p2 = cf3
            nn = index
        else:
            p1 = cf1
            p2 = cf2
            nn = index - 1            
        p3 = np.cos(2 * np.pi / len(data))
        value = (
            (nn / len(data)) + (1/np.pi) * np.arcsin(
                np.sin(2*np.pi/len(data)) * 
                ((-(p1+p2*p3)*(p1-p2) + p2*np.sqrt(p3**2*(p1+p2)**2 - 2*p1*p2*(2*p3**2-p3-1)))/(p1**2 + p2**2 + 2*p1*p2*p3))
            )
        )
    
    return np.absolute(1 - value)



#setting parameters
#scala=100
#epsilon = 1.0

r_1=5. #raggio interno HEL
r_2=2*r_1. #raggio esterno HEL
raggio=6.7
radius=raggio
emittance_star=2.5e-6
beta=280
gamma=(7000/0.938272088)
beta_rel=np.sqrt(1-(1/(gamma**2)))
emittance=emittance_star/(gamma*beta_rel)
th_MAX= 0.3e-6 * 0.0022 / (np.sqrt(beta*emittance)*r_2)
omega_0x= 0.31 * 2*np.pi
omega_1x= -1.73e5 * 2*np.pi * 2*emittance_star/(beta_rel*gamma)
omega_2x= -1.87e12 * 2*np.pi * (2*emittance_star/(beta_rel*gamma))**2
omega_0y= 0.32 * 2 * np.pi
omega_1y= -0.77e5 * 2*np.pi * 2*emittance_star/(beta_rel*gamma)
omega_2y= -3.36e12 * 2*np.pi * (2*emittance_star/(beta_rel*gamma))**2
omega_1xy= 0.92e5 * 2*np.pi * 2*emittance_star/(beta_rel*gamma)
omega_2xy= -1.49e12 * 2*np.pi * (2*emittance_star/(beta_rel*gamma))**2
omega_2yx= 0.66e12 * 2*np.pi * (2*emittance_star/(beta_rel*gamma))**2


iterations=4096
#n=30
n_particles=10**3



#x0=np.linspace(2, (raggio-0.5)/np.sqrt(2), n)
#r=np.random.uniform(r_1, raggio-0.3,n_particles) #*(raggio-0.3)
r=np.full( n_particles, raggio-2)
theta=np.random.rand(n_particles)*np.pi*0.5
x0=r*np.cos(theta)
#x0=np.zeros(n)
#print(x0)
px0=np.zeros(n_particles)
#y0= np.linspace(2, (raggio-0.5)/np.sqrt(2), n)
y0=r*np.sin(theta)
py0=np.zeros(n_particles)



#iterazioni mappa

Xfase1=np.zeros(iterations*n_particles)
Pfase1=np.zeros(iterations*n_particles)
Yfase1=np.zeros(iterations*n_particles)
Pfase1Y=np.zeros(iterations*n_particles)

mappa = sm.generate_instance(omega_0x, omega_1x, omega_2x, omega_0y, omega_1y, omega_2y, omega_1xy, omega_2xy, omega_2yx, 1.0, r_1, r_2, th_MAX, radius, x0, px0, y0, py0)
for i in range(iterations):
    #rumore=np.array([noise[i]])
    mappa.common_noise(make_correlated_noise(1))
    #mappa.compute_personale_noise(1)

    x, px, tx = mappa.get_data()
    y, py, ty = mappa.get_data_y()
    for e in range(len(x)):
        #Xfase1[i*n + e]=x[e]
        #Pfase1[i*n + e]=p[e]
        Xfase1[i+ (iterations * e)]=x[e]
        Pfase1[i+ (iterations * e)]=px[e]
        Yfase1[i+ (iterations * e)]=y[e]
        Pfase1Y[i+ (iterations * e)]=py[e]



#calcolo tune

lista=np.linspace(1, iterations, iterations)
hanning=2*np.sin(lista*np.pi/iterations)**2

tune_x=np.zeros(n_particles)
for k in range(n_particles):
    slice_l = k * (iterations )
    slice_r = (k + 1) * (iterations )

    signal = Xfase1[slice_l: slice_r] + 1j * Pfase1[slice_l: slice_r]
    #fft = np.absolute(np.fft.fft(signal * np.hanning(signal.shape[0])))
    fft = np.absolute(np.fft.fft(signal * hanning)) 
    value = np.argmax(fft)
    value = interpolation(fft, value)
    tune_x[k]=value
tune_y=np.zeros(n_particles)
for k in range(n_particles):
    slice_l = k * (iterations )
    slice_r = (k + 1) * (iterations )

    signal = Yfase1[slice_l: slice_r] + 1j * Pfase1Y[slice_l: slice_r]
   # fft = np.absolute(np.fft.fft(signal * np.hanning(signal.shape[0])))
    fft = np.absolute(np.fft.fft(signal * hanning))
    value = np.argmax(fft)
    value = interpolation(fft, value)
    tune_y[k]=value
    




"""test=np.zeros(n_particles)
testy=np.zeros(n_particles)
for i in range(n_particles):
    action_x=x0[i]**2 *0.5
    action_y=y0[i]**2 *0.5
    test[i]=(omega_0x + (omega_1x * action_x) + (omega_1xy * action_y) + (0.5*  omega_2x * action_x * action_x)+ ( omega_2yx * action_x * action_y)+ (0.5*  omega_2xy * action_y * action_y))/(2*np.pi)
    testy[i]= (omega_0y + (omega_1y * action_y) + (omega_1xy * action_x) + (0.5*  omega_2y * action_y * action_y)+ ( omega_2xy * action_x * action_y)+ (0.5*  omega_2yx * action_x * action_x))/(2*np.pi)
        """


fig, sx=plt.subplots()
fig, ax=plt.subplots()
fig, lx=plt.subplots()
#d=np.linspace(0.306, 0.31, 2)
#dx=np.array([r_1, r_1])
ax.set_ylabel("Q")
ax.set_xlabel("x,y")
#ax.plot(dx, d)
#ax.plot(x0, test)
#ax.plot(y0, testy)
ax.plot(x0, tune_x, "r.", markersize=0.5)
ax.plot(y0, tune_y, "r.", markersize=0.5)

sx.set_ylabel("Qy")
sx.set_xlabel("Qx")
sx.plot(tune_x, tune_y, "r.", markersize=0.5)


#futuro scan di risonanze
"""x0=x0[x!=0]
y0=y0[x!=0]
n=20
m=-10
p=3
bo=n*tune_x+m*tune_y
epsilon=0.01
lx.set_ylabel("y")
lx.set_xlabel("x")
#lx.plot(x0[abs(bo-p)<epsilon], y0[abs(bo-p)<epsilon], "r.", markersize=1)"""
lx.set_ylabel("y")
lx.set_xlabel("x")
lx.plot(x0, y0, "r.", markersize=1)


"""sx.set_xlabel("x")
sx.set_ylabel("P")
sx.plot(Yfase1, Pfase1Y, "r.", markersize=0.5)"""



#plt.xlim(4.2, 6.7)
#plt.ylim(-1.4, 2.3)
plt.show()
