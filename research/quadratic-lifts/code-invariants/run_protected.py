import numpy as np
from protected_span import analyse
def quad(N, pairs):      # helper: S,s for sum of coefficient * z_i z_j terms and linear terms
    S = np.zeros((N, N)); s = np.zeros(N)
    for (i, j, c) in pairs:
        if j is None: s[i] += c
        elif i == j: S[i, i] += 2*c
        else: S[i, j] += c; S[j, i] += c
    return S, s
# (a) quartic oscillator (q,p,y)
analyse("quartic oscillator", lambda z: np.array([z[1], -z[0]*z[2], 2*z[0]*z[1]]), 3,
        {"constraint y - q^2": quad(3, [(2, None, 1), (0, 0, -1)]), "energy p^2/2 + y^2/4": quad(3, [(1, 1, .5), (2, 2, .25)])})
# (b) non-symmetric quartic V = q^4/4 + q^3/3 + q^2/2, lift y = q^2: Vt = y^2/4 + q y/3 + y/2
analyse("non-symmetric quartic", lambda z: np.array([z[1], -(z[2]/3 + 2*z[0]*(z[2]/2 + z[0]/3 + .5)), 2*z[0]*z[1]]), 3,
        {"constraint y - q^2": quad(3, [(2, None, 1), (0, 0, -1)]), "energy": quad(3, [(1, 1, .5), (2, 2, .25), (0, 2, 1/3), (2, None, .5)])})
# (c) mixed example
analyse("mixed (qp)^2/2+(q^2+p^2)/2", lambda z: np.array([z[0]*z[2] + z[1], -z[1]*z[2] - z[0], z[1]**2 - z[0]**2]), 3,
        {"constraint y - qp": quad(3, [(2, None, 1), (0, 1, -1)]), "energy": quad(3, [(0, 0, .5), (1, 1, .5), (2, 2, .5)])})
# (d) paper's example, reduced lift (Y,W,U)=(q^2,p^2,qp)
analyse("paper ex., reduced (Y,W,U)", lambda z: np.array([2*z[2] + 4*z[2]*z[0], -4*z[2]*(z[1] + 2*z[0]), z[1] - 4*z[0]**2]), 3,
        {"Casimir YW - U^2": quad(3, [(0, 1, 1), (2, 2, -1)]), "energy W/2+YW+Y^2": quad(3, [(1, None, .5), (0, 1, 1), (0, 0, 1)])})
# (e) paper's example, Lie-Poisson lift (x1,x2,y11,y12,y22)
def Fe(z):
    x1, x2, a, b, c = z
    return np.array([-2*x2*a - 4*x2*c, 2*x1*c + x1, -4*a*b - 8*b*c, a - 4*c**2, 4*b*c + 2*b])
analyse("paper ex., Lie-Poisson lift", Fe, 5,
        {"y11 - x1^2": quad(5, [(2, None, 1), (0, 0, -1)]), "y12 - x1 x2": quad(5, [(3, None, 1), (0, 1, -1)]),
         "y22 - x2^2": quad(5, [(4, None, 1), (1, 1, -1)])})
# (f) pendulum (s,c,p)
analyse("pendulum (s,c,p)", lambda z: np.array([z[1]*z[2], -z[0]*z[2], -z[0]]), 3,
        {"circle s^2+c^2": quad(3, [(0, 0, 1), (1, 1, 1)]), "energy p^2/2 - c": quad(3, [(2, 2, .5), (1, None, -1)])})
# (g) octic chain (q,p,y1,y2,u): q'=p, p'=-u y2, y1'=2qp, y2'=4up, u'=3 y1 p
analyse("octic chain (q,p,y1,y2,u)", lambda z: np.array([z[1], -z[4]*z[3], 2*z[0]*z[1], 4*z[4]*z[1], 3*z[2]*z[1]]), 5,
        {"y1 - q^2": quad(5, [(2, None, 1), (0, 0, -1)]), "u - q y1": quad(5, [(4, None, 1), (0, 2, -1)]),
         "y2 - y1^2": quad(5, [(3, None, 1), (2, 2, -1)]), "energy p^2/2 + y2^2/8": quad(5, [(1, 1, .5), (3, 3, 1/8)])})
# (h) FPU-beta, N = 3 masses, fixed ends: z = (q1..q3, p1..p3, r0..r3), r_j = (q_{j+1}-q_j)^2
N = 3; D = np.zeros((N+1, N))
for j in range(N+1):
    if j < N: D[j, j] = 1
    if j > 0: D[j, j-1] = -1
def Ffpu(z):
    q, p, r = z[:N], z[N:2*N], z[2*N:]
    d = D@q
    return np.concatenate([p, -D.T@(d + d*r), 2*d*(D@p)])
named = {}
for j in range(N+1):
    S = np.zeros((3*N+1, 3*N+1)); s = np.zeros(3*N+1); s[2*N+j] = 1
    S[:N, :N] = -2*np.outer(D[j], D[j]); named[f"r{j} - d{j}^2"] = (S, s)
S = np.zeros((3*N+1, 3*N+1)); S[N:2*N, N:2*N] = np.eye(N); S[:N, :N] = D.T@D; S[2*N:, 2*N:] = 0.5*np.eye(N+1)
named["energy"] = (S, np.zeros(3*N+1))
analyse("FPU-beta N=3 (q,p,r)", Ffpu, 3*N+1, named)
# (i) Bose-Hubbard dimer in Bloch variables S=(X,Y,Z): H = -2J X - Z^2 (U=1), S' = S x grad H
Jc = 0.5
analyse("BH dimer (Bloch)", lambda S: np.cross(S, np.array([-2*Jc, 0, -2*S[2]])), 3,
        {"Casimir |S|^2": quad(3, [(0, 0, 1), (1, 1, 1), (2, 2, 1)]), "energy -2JX - Z^2": quad(3, [(0, None, -2*Jc), (2, 2, -1)])})
# (j) 2-DOF coupled pendula, constant non-diagonal mass matrix G: z=(pi1,pi2,s1,c1,s2,c2), H = pi'G pi/2 - g1 c1 - g2 c2
G = np.array([[1.0, 0.4], [0.4, 0.8]]); g = [1.0, 0.7]
def Fcp(z):
    pi = z[:2]; s1, c1, s2, c2 = z[2:]; th = G@pi
    return np.array([-g[0]*s1, -g[1]*s2, c1*th[0], -s1*th[0], c2*th[1], -s2*th[1]])
S = np.zeros((6, 6)); S[:2, :2] = G; s = np.zeros(6); s[3] = -g[0]; s[5] = -g[1]
analyse("coupled pendula (mass matrix)", Fcp, 6,
        {"circle 1": quad(6, [(2, 2, 1), (3, 3, 1)]), "circle 2": quad(6, [(4, 4, 1), (5, 5, 1)]), "energy": (S, s)})
