"""The lifted Kahan map of H = |p|^2/2 + U(|q|^2), U(y) = u2 y^2/2 + u1 y, in closed form (O(n) work per step).
Checks against the generic (2n+1)x(2n+1) linear solve, and checks two identities used in the paper:
  q_n ^ q_{n+1} = h M_h  (discrete angular momentum),   q_{n+1} + q_{n-1} = lambda_n q_n  (central recurrence)."""
import numpy as np
rng = np.random.default_rng(1)
n, h, u2, u1 = 4, 0.37, 0.8, -0.3
Up = lambda y: u2*y + u1
def step_closed(q, p, y):
    al = h*Up(y); D = 1 + h*al/2; a, b, c = q@q, q@p, p@p
    yn = (y*D - h*(al + h*u1)*(a + h*b/2) + 2*h*b + h**2*c)/(D + h**2*u2*(a + h*b/2))
    aln = h*Up(yn)
    qn = ((1 - h*aln/2)*q + h*p)/D
    pn = (-(al + aln)*q + (1 - h*al/2)*p)/D
    return qn, pn, yn
def F(z):
    q, p, y = z[:n], z[n:2*n], z[2*n]; return np.concatenate([p, -2*Up(y)*q, [2*q@p]])
def JF(z):
    q, p, y = z[:n], z[n:2*n], z[2*n]; J = np.zeros((2*n + 1, 2*n + 1))
    J[:n, n:2*n] = np.eye(n); J[n:2*n, :n] = -2*Up(y)*np.eye(n); J[n:2*n, 2*n] = -2*u2*q
    J[2*n, :n] = 2*p; J[2*n, n:2*n] = 2*q; return J
def step_generic(z): return z + h*np.linalg.solve(np.eye(2*n + 1) - h/2*JF(z), F(z))
q, p, y = rng.normal(size=n), rng.normal(size=n), rng.normal()
z = np.concatenate([q, p, [y]]); err = 0; Q = [q]; Y = [y]; P = [p]
for k in range(50):
    qn, pn, yn = step_closed(q, p, y); z = step_generic(z)
    err = max(err, np.abs(np.concatenate([qn, pn, [yn]]) - z).max()); q, p, y = qn, pn, yn; Q.append(q); P.append(p); Y.append(y)
print(f"closed-form step vs generic linear solve, 50 steps: max difference {err:.1e}")
wedge = lambda a, b: np.outer(a, b) - np.outer(b, a)
d1 = max(np.abs(wedge(Q[k], Q[k+1]) - h*wedge(Q[k], P[k])/(1 + h**2/2*Up(Y[k]))).max() for k in range(50))
print(f"q_k ^ q_(k+1) = h M_h: max deviation {d1:.1e}")
d2 = 0
for k in range(1, 50):
    lam = (2 - h**2/2*(Up(Y[k+1]) + Up(Y[k-1])))/(1 + h**2/2*Up(Y[k]))
    d2 = max(d2, np.abs(Q[k+1] + Q[k-1] - lam*Q[k]).max())
print(f"q_(k+1) + q_(k-1) = lambda_k q_k with lambda_k = (2 - h^2/2 (U'_(k+1) + U'_(k-1)))/(1 + h^2/2 U'_k): max deviation {d2:.1e}")
