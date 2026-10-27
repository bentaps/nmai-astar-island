"""Does the modified angular momentum extend to coupled lattices? Test: O(2)-symmetric phi^4 lattice with 2 coupled
sites (u_k in R^2, lift y_k = |u_k|^2). The total internal charge Q = sum_k u_k x u_k' is an exact integral of the ODE,
but the block theorem does not apply (the second derivative of Q along the coupling is not an invariant), and
indeed no simple modified version is conserved."""
import numpy as np
from qlift import kahan_step, run
def Fo(z):
    u = z[:4].reshape(2, 2); v = z[4:8].reshape(2, 2); y = z[8:10]
    acc = np.array([(u[1] - u[0]) - u[0] - u[0]*y[0], (u[0] - u[1]) - u[1] - u[1]*y[1]])
    return np.concatenate([v.ravel(), acc.ravel(), [2*u[0]@v[0], 2*u[1]@v[1]]])
I = np.eye(10); JF = lambda z: np.array([(Fo(z + I[k]) - Fo(z - I[k]))/2 for k in range(10)]).T
rng = np.random.default_rng(3); u0 = rng.normal(size=(2, 2))*0.7; v0 = rng.normal(size=(2, 2))*0.5; h = 0.2
Z = run(lambda z: kahan_step(Fo, JF, z, h), np.concatenate([u0.ravel(), v0.ravel(), (u0**2).sum(1)]), 50000)
u = Z[:, :4].reshape(-1, 2, 2); v = Z[:, 4:8].reshape(-1, 2, 2); y = Z[:, 8:10]
Qk = u[:, :, 0]*v[:, :, 1] - u[:, :, 1]*v[:, :, 0]; Q = Qk.sum(1)
print(f"O(2) phi^4, 2 coupled sites, h={h}, 5e4 steps: drift of total charge {abs(Q - Q[0]).max():.1e}; of"
      f" sum_k Q_k/(1 + h^2 (2 + y_k)/2) {abs((Qk/(1 + h**2*(2 + y)/2)).sum(1) - (Qk[0]/(1 + h**2*(2 + y[0])/2)).sum()).max():.1e}")
