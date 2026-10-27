import numpy as np
from qlift import kahan_step, run
def jac(F, N):
    I = np.eye(N)
    return lambda z: np.array([(F(z + I[k]) - F(z - I[k]))/2 for k in range(N)]).T
# ---- open Toda chain, n particles: H = |p|^2/2 + sum_k u_k,  u_k = exp(q_k - q_{k+1});  state (p, u): no constraints
n = 5
def Ft(z):
    p, u = z[:n], z[n:]
    pd = np.zeros(n); pd[:-1] -= u; pd[1:] += u          # p_k' = u_{k-1} - u_k
    return np.concatenate([pd, u*(p[:-1] - p[1:])])
h = 0.2; rng = np.random.default_rng(0)
z0 = np.concatenate([rng.normal(size=n)*0.5, np.exp(rng.normal(size=n-1)*0.3)])
Z = run(lambda z: kahan_step(Ft, jac(Ft, 2*n-1), z, h), z0, 100000)
pd = np.array([Ft(z)[:n] for z in Z])
E = 0.5*np.sum(Z[:, :n]**2, 1) + np.sum(Z[:, n:], 1); Eh = E - h**2/8*np.sum(pd**2, 1)
print(f"open Toda (n={n}, h={h}, 1e5 steps): drift of E - h^2/8|p'|^2 = {abs(Eh - Eh[0]).max():.1e};  plain energy error {abs(E - E[0]).max():.1e}")
# ---- back-substitution: projected (q,p) scheme for H = p^2/2 + q^4/4 must reproduce lifted Kahan from the shifted sheet
F = lambda z: np.array([z[1], -z[0]*z[2], 2*z[0]*z[1]])
h = 0.2; q0, p0 = 1.0, 0.5
Zl = run(lambda z: kahan_step(F, jac(F, 3), z, h), np.array([q0, p0, q0**2 - h**2*p0**2/4]), 2000)
def projected_step(q, p):
    # q' = q + h/2 (p + p');  p' = p - (h/2) q q' (q + q') + (h^3/8)(p'^2 q + p^2 q')   (Newton in p')
    P = p - h*q**3
    for _ in range(50):
        Q = q + h/2*(p + P)
        G = P - p + h/2*q*Q*(q + Q) - h**3/8*(P**2*q + p**2*Q)
        dQ = h/2
        dG = 1 + h/2*q*(dQ*(q + Q) + Q*dQ) - h**3/8*(2*P*q + p**2*dQ)
        dP = -G/dG; P += dP
        if abs(dP) < 1e-16: break
    return q + h/2*(p + P), P
q, p = q0, p0; maxdiff = 0
for k in range(2000):
    q, p = projected_step(q, p); maxdiff = max(maxdiff, abs(q - Zl[k+1, 0]), abs(p - Zl[k+1, 1]))
print(f"back-substituted (q,p) scheme vs lifted Kahan, 2000 steps: max difference {maxdiff:.1e}")
# ---- Morse oscillator H = p^2/2 + D (1 - u)^2,  u = exp(-a (r - r0)):  p' = -2 a D u (1 - u),  u' = -a u p.
# The energy depends on the two variables (p, u) and (p', u') = A J grad E with A = a u, so the two-variable theorem gives
# E~ = (I - h^2 D^2 a^2 u^2/2)/(1 + h^2 D a^2 u^2/2),  I = E - D = p^2/2 + D u^2 - 2 D u   (D1 = 2D, D2 = -4D^2).
Dm, am = 1.3, 0.8
Fm = lambda z: np.array([-2*am*Dm*z[1]*(1 - z[1]), -am*z[1]*z[0]])
h = 0.2; Zm = run(lambda z: kahan_step(Fm, jac(Fm, 2), z, h), np.array([0.9, 1.0]), 100000)
p, u = Zm.T; I = p**2/2 + Dm*u**2 - 2*Dm*u
Et = (I - h**2*Dm**2*am**2*u**2/2)/(1 + h**2*Dm*am**2*u**2/2)
print(f"Morse (h={h}, 1e5 steps): drift of modified energy {abs(Et - Et[0]).max():.1e};  plain energy error {abs(I - I[0]).max():.1e}")
