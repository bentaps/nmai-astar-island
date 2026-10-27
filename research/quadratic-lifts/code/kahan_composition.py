"""Higher order by symmetric composition (Yoshida triple jump) of the lifted Kahan map (quartic oscillator).
Each sub-step preserves ITS OWN modified constraint (h_i-dependent), so the composition preserves none exactly.
Question: order 4? bounded or drifting constraint/energy errors?"""
import numpy as np
from qlift import kahan_step
F = lambda z: np.array([z[1], -z[0]*z[2], 2*z[0]*z[1]])
JF = lambda z: np.array([[0, 1, 0], [-z[2], 0, -z[0]], [2*z[1], 2*z[0], 0]])
H = lambda q, p: p**2/2 + q**4/4
g = 1/(2 - 2**(1/3)); W = [g, 1 - 2*g, g]
def yoshida(z, h):
    for w in W: z = kahan_step(F, JF, z, w*h)
    return z
def rk4(q0, p0, T, n):
    f = lambda z: np.array([z[1], -z[0]**3]); z = np.array([q0, p0]); h = T/n
    for _ in range(n):
        k1 = f(z); k2 = f(z+h/2*k1); k3 = f(z+h/2*k2); k4 = f(z+h*k3); z = z + h/6*(k1+2*k2+2*k3+k4)
    return z
q0, p0 = 1.0, 0.5; ref = rk4(q0, p0, 10.0, 200000); prev = None
for h in [0.2, 0.1, 0.05, 0.025]:
    z = np.array([q0, p0, q0**2])
    for _ in range(int(round(10/h))): z = yoshida(z, h)
    e = np.hypot(z[0]-ref[0], z[1]-ref[1])
    print(f"h={h:<6} error {e:.2e}" + ("" if prev is None else f"  rate {np.log2(prev/e):.2f}")); prev = e
h = 0.1; z = np.array([q0, p0, q0**2]); n = 200000
cons, en = np.empty(n), np.empty(n)
for k in range(n):
    z = yoshida(z, h); cons[k] = z[2] - z[0]**2; en[k] = H(z[0], z[1]) - H(q0, p0)
for a, b in [(0, n//100), (n - n//100, n)]:
    print(f"steps {a}-{b}: max|y-q^2| = {np.max(abs(cons[a:b])):.2e},  max|H-H0| = {np.max(abs(en[a:b])):.2e}")
