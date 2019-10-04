import numpy as np
import matplotlib.pyplot as plt

xs = np.linspace(-1,1,100)
def fun(xs):
    #transform xs to change -1,1 to 0.5,1
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
    for order in range(1,max_order):
        coefs = lin_fit(cheby_mat(xs, order),ys)

        error = np.abs(coefs[-1])
        print("Order %d gives error %.7f" % (order - 1, error))
        if error < tol:
            return coefs[:-1], order-1

    print("MAX ORDER REACHED, DO NOT TRUST RESULTS")
    return coefs, order

def poly_fit(xs,ys,order):
    return lin_fit(poly_mat(xs, order), ys)

def lin_fit(A, ys):
    lhs = A.transpose() @ A
    rhs = A.transpose() @ ys
    return np.linalg.inv(lhs) @ rhs

def log2(x, cheby_coef):
    mantissa, exp = np.frexp(x)
    log_mantissa = cheby_mat(mantissa * 4 - 3, len(cheby_coef)-1) @ cheby_coef
    return log_mantissa + exp

if __name__ == "__main__":
    # Fitted Data
    cheby_coef, order = cheby_fit(xs,ys)
    poly_coef = poly_fit(xs, ys, order)

    # Plotting Results
    real_xs = np.linspace(0.5,1,len(xs))
    fig, axes = plt.subplots(2,1, sharex=True, dpi=255);
    axes[0].set_title("Cheby and Poly fit of log_2(x)")

    # Plot Data
    axes[0].scatter(real_xs, ys, label="Real Data", color="gray", marker="x")

    cheby_data = cheby_mat(xs,order) @ cheby_coef
    axes[0].plot(real_xs, cheby_data, linestyle="--", label="Cheby")

    poly_data = poly_mat(xs, order) @ poly_coef
    axes[0].plot(real_xs, poly_data, linestyle="-.", label="Poly")
    axes[0].set_ylabel("log_2(x)")
    # Plot Residuals
    axes[1].hlines(0,0.5,1)
    cheby_res = ys - cheby_data
    axes[1].scatter(real_xs, ys - cheby_data, label="Cheby")
    poly_res = ys - poly_data
    axes[1].scatter(real_xs, ys - poly_data, label="Poly")
    axes[1].set_ylim([min(cheby_res.min(), poly_res.min()) * 1.1,
                      max(cheby_res.max(), poly_res.max()) * 1.1])
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("Residual")
    axes[0].legend()
    axes[1].legend()


    # Output Error Calculations
    print("Maximum Errors:")
    print("\t Cheby: %e" % (max(cheby_res)))
    print("\t Poly: %e" % (max(poly_res)))
    print("RMS Errors:")
    print("\t Cheby: %e" % np.sqrt(np.sum(np.power(cheby_res,2))/len(xs)))
    print("\t Poly: %e" % np.sqrt(np.sum(np.power(poly_res,2))/len(xs)))

    plt.show(block=True)

    # Plotting log2 over a larger range

    xs = np.linspace(1E-3, 1E3, 1000)
    ys = log2(xs, cheby_coef)
    real_ys = np.log2(xs)
    res = real_ys - ys

    fig, axes = plt.subplots(2,1, sharex=True, dpi=255);
    axes[0].set_title("Arbitrary log from Cheby")

    # Plot Data
    axes[0].plot(xs, real_ys, linestyle="-.", label="Numpy")
    axes[0].plot(xs, ys, linestyle="--", label="Cheby")
    axes[0].set_ylabel("log_2")
    axes[0].legend()

    # Plot Residuals
    axes[1].hlines(0,1E-3,1E3)
    axes[1].scatter(xs, res)
    axes[1].set_ylim([-1E-6,1E-6])
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("Residual")

    plt.show(block=True)
