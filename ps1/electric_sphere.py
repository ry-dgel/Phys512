import integrate_vary_step as ivs
import scipy.integrate as itg
from scipy.constants import epsilon_0
import numpy as np
import matplotlib.pyplot as plt

# Electric field of ring with charge Q, radius R at point
# along central axis a distance z away
def ring(Q, R, z):
    k = 1.0/(4 * np.pi * epsilon_0)
    return (k * Q * z) / (z**2 + R**2) ** (3/2)

# radius of a disc containing the point a distance z along a sphere's radius R
def rad(R,z):
    return np.sqrt(R**2 - z**2)

# The electric field contribution of a ring a height z above a spheres
# center at a point r away from the center of the sphere, for a sphere with
# charge Q and radius R.
def sphere_seg(Q, R, r, z):
    ring_rad = rad(R,z)
    # Charge of ring becomes Q/R**@ as this is the differential charge density.
    return ring(Q/R**2,ring_rad,r-z)

# Integration of sphere using my integral method
def sphere_mine(Q,R,r):
    fun = lambda z: sphere_seg(Q,R,r,z)

    val,_,_ = ivs.integrate(fun, -R, R, 1E-6)
    return val

def sphere_scip(Q,R,r):
    fun = lambda z: sphere_seg(Q,R,r,z)
    res = itg.quad(fun, -R, R, epsrel=1E-6, points=[R,-R])
    val = res[0]
    return val

if __name__ == "__main__":
    
    # Insert sensible parameters here
    R = 1
    Q = 10E-6
    rs = np.linspace(0,5,1000)

    # Very pythonian way of avoiding the singularities...
    Emine = []
    for r in rs:
        try:
            Emine.append(sphere_mine(Q,R,r))
        except RecursionError:
            Emine.append(np.inf)

    # We trust that scipy knows what it's doing
    Escip = [sphere_scip(Q,R,r) for r in rs]
    
    # Plot the stuff
    plt.plot(rs,Emine)
    plt.plot(rs,Escip,linestyle='--')
    plt.xlabel("Distance from center of sphere")
    plt.ylabel("Electric field amplitude")
    # Wait until plot closes to quit
    plt.show(block=True)
