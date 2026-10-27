"""The prolongation trick: for a quadratic ODE x' = f(x), add v = x' as a variable: x' = v (affine), v' = f'(x) v.
The constraint v - f(x) is an exact integral whose factors (x) move affinely, so Kahan conserves
c_h = v - f(x) + (h^2/8) D^2 f(x)[v, v]  exactly (companion lemma). But the original quadratic integrals are only
conditional for the prolonged system, so they are not protected. Test: Euler top x' = (a1 x2 x3, a2 x3 x1, a3 x1 x2)."""
import numpy as np
from qlift import kahan_step, run
a = np.array([1.0, -2.0, 1.0])                             # a1 + a2 + a3 = 0: integrals |x|^2 and sum x_i^2/a_i-type
f = lambda x: a*np.array([x[1]*x[2], x[2]*x[0], x[0]*x[1]])
def Df(x): return np.array([[0, a[0]*x[2], a[0]*x[1]], [a[1]*x[2], 0, a[1]*x[0]], [a[2]*x[1], a[2]*x[0], 0]])
def D2f(v):                                                # D^2 f[v, v] (constant Hessians)
    return 2*a*np.array([v[1]*v[2], v[2]*v[0], v[0]*v[1]])
def F(z): x, v = z[:3], z[3:]; return np.concatenate([v, Df(x)@v])
I6 = np.eye(6); JF = lambda z: np.array([(F(z + I6[k]) - F(z - I6[k]))/2 for k in range(6)]).T
h = 0.1; x0 = np.array([0.6, 0.3, -0.5])
# start on the shifted sheet: solve v = f(x) - h^2/8 D^2 f[v, v] by fixed-point iteration
v = f(x0)
for _ in range(100): v = f(x0) - h**2/8*D2f(v)
Z = run(lambda z: kahan_step(F, JF, z, h), np.concatenate([x0, v]), 20000)
ch = np.array([z[3:] - f(z[:3]) + h**2/8*D2f(z[3:]) for z in Z])
I1 = (Z[:, :3]**2).sum(1)                                  # exact integral of the Euler top (here a1 + a2 + a3 = 0)
print(f"prolonged Kahan, Euler top, 2e4 steps: max |c_h| = {np.abs(ch).max():.1e} (conserved);"
      f"  drift of |x|^2 = {abs(I1 - I1[0]).max():.1e} (an integral of the ODE, not protected after prolongation)")
# compare: plain Kahan on the Euler top conserves a modified |x|^2 (two-variable integrals; Hirota-Kimura)
Zp = run(lambda x: kahan_step(f, Df, x, h), x0, 20000); I1p = (Zp**2).sum(1)
print(f"plain Kahan on the Euler top: drift of |x|^2 = {abs(I1p - I1p[0]).max():.1e} (bounded; its modified version is exact)")
