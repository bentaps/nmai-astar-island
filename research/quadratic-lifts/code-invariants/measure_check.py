import numpy as np
def check(F, JF, P, h, label, n=400):
    rng = np.random.default_rng(1); worst = 0
    for _ in range(n):
        z = rng.uniform(-1, 1, 3)
        M = np.eye(3) - h/2*JF(z); zn = z + h*np.linalg.solve(M, F(z))
        det = np.linalg.det(np.linalg.solve(M, np.eye(3) + h/2*JF(zn)))
        worst = max(worst, abs(P(zn) - det*P(z))/abs(P(z)))
    print(f"{label}: max relative defect of P(z') = det(DPhi) P(z): {worst:.1e}")
h = 0.2
F = lambda z: np.array([z[0]*z[2] + z[1], -z[1]*z[2] - z[0], z[1]**2 - z[0]**2])
JF = lambda z: np.array([[z[2], 1, z[0]], [-1, -z[2], -z[1]], [-2*z[0], 2*z[1], 0]])
check(F, JF, lambda z: (1 + h**2*(z[0]+z[1])**2/4)*(1 + h**2*(z[0]-z[1])**2/4), h, "mixed: P = (1 + h^2 u^2/2)(1 + h^2 v^2/2)")
check(F, JF, lambda z: (1 + h**2*(z[0]+z[1])**2/4), h, "mixed: P = (1 + h^2 u^2/2) alone")
F2 = lambda z: np.array([z[1], -(z[2]/3 + 2*z[0]*(z[2]/2 + z[0]/3 + .5)), 2*z[0]*z[1]])
JF2 = lambda z: np.array([[0, 1, 0], [-(2*(z[2]/2 + z[0]/3 + .5) + 2*z[0]/3), 0, -(1/3 + z[0])], [2*z[1], 2*z[0], 0]])
check(F2, JF2, lambda z: 1 + h**2/4*2*z[0]*(z[0] + 1/3), h, "non-symmetric quartic: P = 1 + (h^2/4) 2q(q + 1/3)")
