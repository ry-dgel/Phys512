import numpy as np
import matplotlib.pyplot as plt

xs = np.arange(-1,1,0.1)
def fun(xs):
    #transform xs to change 0.5,1 to -1,1
    return np.log2((xs + 3)/4)
ys = fun(xs)

def cheby_mat(xs,order):
    assert order >= 0, "Order must be positive"
    assert min(xs) >= -1 and max(xs) <= 1, "x out of bounds"
    mat = np.ones((len(xs),order+1))
    if order > 0:
        mat[:,1] = xs
    for i in range(1,order):
        mat[:,i+1] = 2*xs*mat[:,i] - mat[:,i-1]
    return mat

def poly_mat(xs,order):
    assert order >= 0, "Order must be positive"
    assert min(xs) >= -1 and max(xs) <= 1, "x out of bounds"
    mat = np.ones((len(xs),order+1))
    if order > 0:
        mat[:,1] = xs
    for i in range(1,order):
        mat[:,i+1] = xs * mat[:,i]
    return mat

def cheby_fit(xs,ys,tol=1E-6, max_order=10):
    for order in range(max_order):
        coefs = lin_fit(cheby_mat(xs, order),ys)
        
        error = np.abs(coefs[-1])
        # print("Order %d gives error %.7f" % (order, error))
        # print(coefs)
        # plt.plot((xs + 3)/4,ys)
        # plt.plot((xs + 3)/4,A@coefs)
        # plt.show(block=True)
        if error < tol:
            return coefs, order

    print("MAX ORDER REACHED, DO NOT TRUST RESULTS")
    return coefs, order

def poly_fit(xs,ys,order):
    return lin_fit(poly_mat(xs, order), ys)

def lin_fit(A, ys):
    lhs = A.transpose() @ A
    rhs = A.transpose() @ ys
    return np.linalg.inv(lhs) @ rhs

cheby_coef, order = cheby_fit(xs,ys)
poly_coef = poly_fit(xs, ys, order)
