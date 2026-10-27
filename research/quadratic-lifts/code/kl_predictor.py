"""Kahan-lift as predictor for the MQAV midpoint Newton iteration (phi^4 KG)."""
import numpy as np, scipy.sparse as sps, scipy.sparse.linalg as spla
from kg_pde import setup, energy, KL_step
N, L = 512, 20.0
x, dx, D2 = setup(N, L); I = sps.identity(N, format='csc')
u0 = 2.0/np.cosh(x); v0 = np.zeros(N); E0 = energy(u0, v0, dx, D2)

def mqav_newton(u, v, h, un, tol, maxit):
    for it in range(maxit):
        ub = 0.5*(u + un)
        G = (2/h**2)*(un - u - h*v) - D2@ub + ub + ub*(u**2 + un**2)/2
        J = (2/h**2)*I - 0.5*D2 + 0.5*I + sps.diags(0.5*(u**2 + un**2)/2 + ub*un)
        d = spla.spsolve(J.tocsc(), -G); un = un + d
        if np.max(abs(d)) < tol*(1 + np.max(abs(un))): return un, it + 1
    return un, maxit

for ratio in [0.5, 4.0]:
    h = ratio*dx; n = int(round(100/h))
    for label, pred, maxit, tol in [("MQAV, Euler predictor, tol 1e-12", 'euler', 30, 1e-12),
                                    ("MQAV, KL predictor,    tol 1e-12", 'kl', 30, 1e-12),
                                    ("KL + 1 Newton correction        ", 'kl', 1, 0.0)]:
        u, v = u0.copy(), v0.copy(); its = 0; maxdE = 0
        for k in range(n):
            if pred == 'euler': guess = u + h*v
            else: guess = KL_step(u, v, u**2, h, D2, I)[0]; its += 1
            un, it = mqav_newton(u, v, h, guess, tol, maxit); its += it
            v = 2*(un - u)/h - v; u = un
            maxdE = max(maxdE, abs(energy(u, v, dx, D2) - E0)/E0)
        print(f"h={ratio}dx  {label}: sparse solves/step {its/n:.2f},  max rel. energy error (T=100) {maxdE:.1e}")
