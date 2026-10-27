import numpy as np
from qlift import kahan_step, run, algebraic_relation_spectrum
def jac(F, N):
    I = np.eye(N)
    return lambda z: np.array([(F(z + I[k]) - F(z - I[k]))/2 for k in range(N)]).T
# ---- coupled pendula with constant mass matrix G: z=(pi1,pi2,s1,c1,s2,c2)
G = np.array([[1.0, 0.4], [0.4, 0.8]]); g = np.array([1.0, 0.7]); h = 0.15
def Fcp(z):
    pi = z[:2]; s1, c1, s2, c2 = z[2:]; th = G@pi
    return np.array([-g[0]*s1, -g[1]*s2, c1*th[0], -s1*th[0], c2*th[1], -s2*th[1]])
q0 = np.array([1.2, -0.7]); z = np.array([0.3, 0.5, np.sin(q0[0]), np.cos(q0[0]), np.sin(q0[1]), np.cos(q0[1])])
Z = run(lambda z: kahan_step(Fcp, jac(Fcp, 6), z, h), z, 200000)
th = Z[:, :2]@G.T; pid = -np.column_stack([g[0]*Z[:, 2], g[1]*Z[:, 4]])
K1 = (Z[:, 2]**2 + Z[:, 3]**2)/(1 + h**2*th[:, 0]**2/4); K2 = (Z[:, 4]**2 + Z[:, 5]**2)/(1 + h**2*th[:, 1]**2/4)
E = 0.5*np.einsum('ni,ij,nj->n', Z[:, :2], G, Z[:, :2]) - g[0]*Z[:, 3] - g[1]*Z[:, 5]
Eh = E - h**2/8*np.einsum('ni,ij,nj->n', pid, G, pid)
Etrue = 0.5*np.einsum('ni,ij,nj->n', Z[:, :2], G, Z[:, :2]) - g[0]*np.cos(np.arctan2(Z[:, 2], Z[:, 3])) - g[1]*np.cos(np.arctan2(Z[:, 4], Z[:, 5]))
n = len(E)//10
print(f"coupled pendula (2e5 steps): drift K1_h {abs(K1-K1[0]).max():.1e}, K2_h {abs(K2-K2[0]).max():.1e}, E_h {abs(Eh-Eh[0]).max():.1e};"
      f" true energy error first/last 10%: {abs(Etrue[:n]-Etrue[0]).max():.2e} / {abs(Etrue[-n:]-Etrue[0]).max():.2e}")
# ---- Bose-Hubbard dimer, Bloch S, H = -2J X - Z^2: S' = S x gradH
Jc = 0.5; FB = lambda S: np.cross(S, np.array([-2*Jc, 0, -2*S[2]]))
Z = run(lambda z: kahan_step(FB, jac(FB, 3), z, 0.2), np.array([0.3, 0.2, 0.9]), 3000)
print("BH dimer: exact relations at deg 2,3,4 =", [int(np.sum(algebraic_relation_spectrum(Z, d)[0] < 1e-11)) for d in (2, 3, 4)], "(two quadrics: [2, 8, 19])")
# ---- octic chain: energy protected but u - q y1, y2 - y1^2 are NOT exact integrals -> drift?
Fo = lambda z: np.array([z[1], -z[4]*z[3], 2*z[0]*z[1], 4*z[4]*z[1], 3*z[2]*z[1]])
q0, p0 = 0.9, 0.3; z0 = np.array([q0, p0, q0**2, q0**4, q0**3]); h = 0.05
Z = run(lambda z: kahan_step(Fo, jac(Fo, 5), z, h), z0, 400000)
q, p, y1, y2, u = Z.T; n = len(q)//10
Eh = (p**2/2 + y2**2/8)/(1 + h**2*u**2)
for lab, c in [("y1 - q^2", y1 - q**2), ("u - q y1", u - q*y1), ("y2 - y1^2", y2 - y1**2), ("true H(q,p) - H0", p**2/2 + q**8/8 - (p0**2/2 + q0**8/8))]:
    print(f"octic chain h={h}: {lab:18s} max|.| first/mid/last 10%: {abs(c[:n]).max():.2e} / {abs(c[4*n:5*n]).max():.2e} / {abs(c[-n:]).max():.2e}")
print(f"octic chain: drift of protected modified energy (p^2/2 + y2^2/8)/(1 + h^2 u^2): {abs(Eh-Eh[0]).max():.1e}")
