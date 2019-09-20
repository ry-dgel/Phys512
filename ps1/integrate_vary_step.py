import numpy as np

sig=0.1

def fun(x):
    return 1.0/(1.0+x**2)

def fun2(x):
    return 1.0+np.exp(-0.5*x**2/(sig**2))

def simple_integrate(fun,a,b,tol):
    x=np.linspace(a,b,5)
    dx=(b-a)/4.0
    #np.median(np.diff(x))
    y=fun(x)
    neval=len(x) #let's keep track of function evaluations
    f1=(y[0]+4*y[2]+y[4])/6.0*(b-a)
    f2=(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/12.0*(b-a)
    myerr=np.abs(f2-f1)
    #print([a,b,f1,f2])
    if (myerr<tol):
        #return (f2)/1.0,myerr,neval
        return (16.0*f2-f1)/15.0,myerr,neval
    else:
        mid=0.5*(b+a)
        f_left,err_left,neval_left=simple_integrate(fun,a,mid,tol/2.0)
        f_right,err_right,neval_right=simple_integrate(fun,mid,b,tol/2.0)
        neval=neval+neval_left+neval_right
        f=f_left+f_right
        err=err_left+err_right
        return f,err,neval

def integrate(fun, a, b, tol, ys=np.array([])):

    # Avoid computation if endpoints are identical.
    if a == b:
        return 0.0,0.0,0.0

    # Compute new function points
    x = np.linspace(a,b,5)
    if ys.size == 0:
        # Otherwise, compute all 5 points (first iteration)
        y = fun(x)
        neval = 5
    else:
        # If old points were provided, only compute two new ones
        y = np.array([ys[0], fun(x[1]), ys[1], fun(x[3]), ys[2]])
        neval = 2

    # Evaluate integrals
    f1 = (y[0] + 4*y[2] + y[4])/6.0 * (b - a)
    f2 = (y[0] + 4*y[1] + 2*y[2] + 4*y[3] + y[4])/12.0 * (b - a)

    # Check errors and return if good enough.
    err = np.abs(f2-f1)
    #print([a,b,f1,f2])
    if (err < tol):
        return (16 * f2 - f1)/15.0, err, neval

    # Compute integrals on smaller sections, passing pre computed points.
    mid = 0.5 * (b+a)
    left, err_l, neval_l  = integrate(fun, a, mid, tol/2.0, y[0:3])
    right, err_r, neval_r = integrate(fun, mid, b, tol/2.0, y[2:])

    # Sum results
    neval = neval + neval_l + neval_r
    f = left + right
    err = err_l + err_r

    # Return
    return f, err, neval

if __name__ == "__main__":
    pred=np.exp(1)-np.exp(-1)
    pred = np.pi/4
    f,err,neval=simple_integrate(fun,0,1,1e-7)
    print("")
    fn,errn,nevaln = integrate(fun,0,1,1e-7)
    print("")
    #f,err,neval=simple_integrate(fun,-1,1,1e-4);pred=np.arctan(1)-np.arctan(-1)
    #a=-5;b=5;f,err,neval=simple_integrate(fun2,a,b,1e-4);pred=(b-a)+np.sqrt(2*np.pi)*sig
    print('f,err,neval are ' + repr([f,err,neval])+' with err ' + repr(np.abs(f-pred)))
    print('New f,err,neval are ' + repr([fn,errn,nevaln])+' with err ' + repr(np.abs(fn-pred)))
