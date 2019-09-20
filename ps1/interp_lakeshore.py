import numpy as np
from sys import argv
import scipy.interpolate as interp

hardcoded_voltage = 0.69

data = np.genfromtxt("lakeshore.txt", names=True)
# Data is given as V(T) in ascending temperature/descending voltage
# We want T(V) and so voltage needs to be ascending for spline to work.
temps = np.flip(data['T'])
fourth_deriv = np.gradient(np.gradient(np.gradient(np.gradient(temps))))
volts = np.flip(data['V'])
spl = interp.splrep(volts, temps)

def cubic_interp(temps, volts, fourth_deriv, voltage):
    # Input voltage must be within bounds of known data to do a cubic fit.
    # I don't want to do polynomial extrapolation...
    # 1 smaller on each side so that we can do error analysis
    assert voltage < volts[-3] and voltage > volts[2], "Invalid input voltage range %f" % voltage
    
    # Find the index of the known value closest to the input voltage to use
    # as our central point
    i = np.argmin(np.abs(volts-voltage))
    # Cubic fit with the neighbouring 4
    poly = np.polyfit(volts[i-1:i+3], temps[i-1:i+3],3)

    #This method doesn't work well:
    #error = max(fourth_deriv[i-1:i+3]) / 24 * np.prod(voltage - volts[i-1:i+3]) 
    
    value = np.polyval(poly, voltage)

    #This one is better:
    left_poly = np.polyfit(volts[i-2:i+2], temps[i-2:i+2],3)
    right_poly = np.polyfit(volts[i:i+4], temps[i:i+4],3)
    error = max(value - np.polyval(left_poly, voltage), 
                value - np.polyval(right_poly,voltage))

    # Return the interpolated value
    return value, error

def print_interp(voltage):


    print("For input voltage %f V, temperatures is:" % voltage)
    try:
        poly, error= cubic_interp(temps,volts,fourth_deriv,voltage)
        print("\tCubic Poly gives: %f ± %f K" % (poly, error))
    except AssertionError:
        print("\tGiven voltage range is invalid for polynomial interpolation")

if len(argv) > 1:
    for inpt in argv[1:]:
        print_interp(float(inpt))
else:
    print_interp(hardcoded_voltage)


