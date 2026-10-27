import numpy as np
rng = np.random.default_rng(1)
N = 7
L = rng.normal(size=(3, N)) * 0.5; l = rng.normal(size=3) * 0.5
Qb = rng.normal(size=(2, N, N)) * 0.3; Qb = 0.5 * (Qb + Qb.transpose(0, 2, 1)); Bb = rng.normal(size=(2, N)) * 0.3
e = np.eye(N)
def f(z):
    a = z[:3]; da = L @ z + l
    db = np.array([z @ Qb[k] @ z for k in range(2)]) + Bb @ z
    return np.concatenate([da, [da[0]*a[1] + a[0]*da[1], 2*a[2]*da[2]], db])
def jac(z):
    a = z[:3]; da = L @ z + l
    J = np.zeros((N, N)); J[:3] = L
    J[3] = L[0]*a[1] + da[0]*e[1] + a[0]*L[1] + da[1]*e[0]
    J[4] = 2*da[2]*e[2] + 2*a[2]*L[2]
    J[5] = 2*Qb[0] @ z + Bb[0]; J[6] = 2*Qb[1] @ z + Bb[1]
    return J
kahan = lambda z, h: z + h*np.linalg.solve(np.eye(N) - 0.5*h*jac(z), f(z))
for h in [0.1, 0.05]:
    z = rng.normal(size=N)*0.3; z[3] = z[0]*z[1]; z[4] = z[2]**2
    I12 = lambda z: z[3] - z[0]*z[1] + h**2/4*f(z)[0]*f(z)[1]
    I33 = lambda z: z[4] - z[2]**2 + h**2/4*f(z)[2]**2
    i0, j0 = I12(z), I33(z); rel = 0; plain = 0
    for k in range(60):
        z = kahan(z, h)
        scale = 1 + np.max(abs(z))**2 + h**2*np.max(abs(f(z)))**2
        rel = max(rel, abs(I12(z)-i0)/scale, abs(I33(z)-j0)/scale); plain = max(plain, abs(z[3]-z[0]*z[1]))
    print(f"h={h}: max relative change of modified integrals = {rel:.1e};  plain |y12-a1a2| reached {plain:.1e};  |z|max={np.max(abs(z)):.2f}")
