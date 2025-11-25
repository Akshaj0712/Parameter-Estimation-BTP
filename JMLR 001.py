import numpy as np
import cmath
import matplotlib.pyplot as plt

def compute_C_params_from_poly(coeffs, alpha, m=4096, eps_R=1e-6, beta=0.2, plot=False, verbose=False):
    """
    coeffs: polynomial coefficients highest-first for p(z) (monic not required)
    alpha: radius (the |z|=alpha circle)
    m: number of sample points on circle
    eps_R: small margin added/subtracted from min/max real-part to ensure strictness
    beta: choose tau0 = L + beta*(U-L) inside feasible interval
    Returns: (feasible, tau0, tau1, tau2, info)
    """
    p = np.asarray(coeffs, dtype=np.complex128)
    n = len(p) - 1
    W = []

    thetas = np.linspace(0, 2*np.pi, m, endpoint=False)
    r = np.empty(m)
    s = np.empty(m)

    for i, th in enumerate(thetas):
        z = alpha * cmath.exp(1j*th)
        pz = np.polyval(p, z)
        # compute w = p(z)/z^n
        w = pz / (z**n)
        W.append(w)
        r[i] = w.real
        s[i] = w.imag
    W = np.array(W)

    # --- Plot P(z)/z^n on complex plane ---
    if plot:
        plt.figure(figsize=(6,6))
        plt.plot(W.real, W.imag, 'b.', markersize=2, label=r'$p(z)/z^n$')
        plt.axhline(0, color='gray', linewidth=0.5)
        plt.axvline(0, color='gray', linewidth=0.5)
        plt.xlabel("Re")
        plt.ylabel("Im")
        plt.title(r"Complex map of $p(z)/z^n$ for $|z|=\alpha$")
        plt.gca().set_aspect('equal', 'box')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.show()
    # tau1, tau2 (strict by subtracting/adding eps_R)
    tau1 = r.min() - eps_R
    tau2 = r.max() + eps_R

    # Compute feasible interval for tau0
    ub_list = []
    lb_list = []
    # track s==0 issues
    s_zero_bad = False
    for rk, sk in zip(r, s):
        if abs(sk) < 1e-14:
            if rk < 0:
                s_zero_bad = True
                if verbose:
                    print("Found s≈0 with r<0 -> wedge impossible for any finite tau0.")
            # else no constraint on tau0
            continue
        ratio = (rk / sk) - 1.0
        if sk > 0:
            ub_list.append(ratio)   # tau0 <= ratio
        else:
            lb_list.append(ratio)   # tau0 >= ratio

    if ub_list:
        U = min(ub_list)
    else:
        U = np.inf  # no upper bound from positive-imag points

    if lb_list:
        L = max(lb_list)
    else:
        L = -np.inf  # no lower bound from negative-imag points

    feasible = (not s_zero_bad) and (L <= U)
    if not feasible:
        return False, None, None, None, {
            'alpha': alpha, 'min_r': float(r.min()), 'max_r': float(r.max()),
            'L': L, 'U': U, 's_zero_bad': s_zero_bad
        }

    # pick tau0 inside [L,U]; use midpoint or conservative bias beta in [0,1]
    if np.isfinite(L) and np.isfinite(U):
        tau0 = L + beta * (U - L)
    elif np.isfinite(L):
        # only lower bound -> choose tau0 = L + 1.0 (some margin)
        tau0 = L + 1.0
    elif np.isfinite(U):
        tau0 = U - 1.0
    else:
        # unconstrained tau0: pick 0
        tau0 = 0.0

    # final verification on samples
    ok = True
    viols = 0
    for rk, sk in zip(r, s):
        if not (rk >= (1.0 + tau0) * sk - 1e-12):  # small tolerance
            ok = False
            viols += 1
        if not (tau1 < rk < tau2):
            ok = False
            viols += 1
    if not ok and verbose:
        print(f"Warning: {viols} sample points violate the chosen parameters (maybe tighten eps or increase m).")

    info = {'alpha': alpha, 'min_r': float(r.min()), 'max_r': float(r.max()), 'L': float(L), 'U': float(U)}
    return True, float(tau0), float(tau1), float(tau2), info

import numpy as np

def coeffs_from_roots(roots):
    """
    roots: iterable of complex or real roots
    returns: numpy array of polynomial coefficients (highest degree first)
    """
    return np.poly(roots)

if __name__ == "__main__":
    # example monic p(z) = (z-0.2)(z-0.3)(z-0.5)
    # Example:
    roots = [0.2, 0.4, 0.3]
    coeffs = coeffs_from_roots(roots)
    coeffs = [1,0,-0.01]
    # coeffs = [1, 0, 0, 0, -25/9/16,0,0,1/9/16]
    import numpy as np

    # p1 = np.array([1, 2, 3])   # 1*z^2 + 2*z + 3
    # p2 = np.array([1, -4, 2])  # 1*z^2 - 4*z + 2

    # p = np.polymul(p1, p2)

    
    print(coeffs)
    alpha = 0.5
    feasible, tau0, tau1, tau2, info = compute_C_params_from_poly(coeffs, alpha, m=8192, eps_R=1e-5, beta=0.3, plot=True)
    print("feasible:", feasible)
    print("tau0, tau1, tau2:", tau0, tau1, tau2)
    print("info:", info)
