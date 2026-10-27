"""Do lifted Kahan maps preserve a measure dz/P with P polynomial? Exact search modulo a prime, generic step size.
Expected output: densities exist for the rotational quartic (2), open Toda (1, the Liouville density u1 u2) and
random 3D Nambu systems (1 each); none for the phi^4 and FPU lattices (up to degree 6) or the 2-site discrete NLS lift.
The FPU degree-6 search takes about 5 minutes."""
import sympy as sp, random, time
from darboux_modp import density_nullity
R = sp.Rational

def report(name, F, vars, degs):
    for d in degs:
        t = time.time(); nul, nm = density_nullity(F, vars, d)
        print(f"{name:50s} degree <= {d}: {nul} density/densities ({nm} monomials, {time.time()-t:.0f}s)", flush=True)
        if nul: break

q1, q2, p1, p2, y = sp.symbols('q1 q2 p1 p2 y')
report("isotropic quartic |q|^4/4, lift y=|q|^2 (5D)", [p1, p2, -q1*y, -q2*y, 2*(q1*p1 + q2*p2)], [q1, q2, p1, p2, y], (2, 4))
a1, a2, a3, u1, u2 = sp.symbols('a1 a2 a3 u1 u2')
report("open Toda n=3, (p, u) variables (5D)", [-u1, u1 - u2, u2, u1*(a1 - a2), u2*(a2 - a3)], [a1, a2, a3, u1, u2], (2,))
x, yy, zz = sp.symbols('x yy zz'); rnd = random.Random(3)
mons = [x**2, yy**2, zz**2, x*yy, x*zz, yy*zz, x, yy, zz]
for t in range(2):
    H = sum(R(rnd.randint(-5, 5), rnd.randint(1, 4))*m for m in mons); C = sum(R(rnd.randint(-5, 5), rnd.randint(1, 4))*m for m in mons)
    gH = sp.Matrix([sp.diff(H, v) for v in (x, yy, zz)]); gC = sp.Matrix([sp.diff(C, v) for v in (x, yy, zz)])
    report(f"random 3D Nambu grad H x grad C, trial {t}", list(gH.cross(gC)), [x, yy, zz], (2, 4))
A1, B1, A2, B2, r1, r2 = sp.symbols('A1 B1 A2 B2 r1 r2'); k = 2
report("2-site discrete NLS, lift rho_k=|psi_k|^2 (6D)",
       [(B1 - B2) + k*r1*B1, -(A1 - A2) - k*r1*A1, (B2 - B1) + k*r2*B2, -(A2 - A1) - k*r2*A2, 2*(A2*B1 - A1*B2), 2*(A1*B2 - A2*B1)],
       [A1, B1, A2, B2, r1, r2], (2, 4))
y1, y2 = sp.symbols('y1 y2'); kc = R(1, 2)
report("phi^4 lattice, 2 sites, lift y_k=q_k^2 (6D)",
       [p1, p2, -kc*(q1 - q2) - q1 - q1*y1, -kc*(q2 - q1) - q2 - q2*y2, 2*q1*p1, 2*q2*p2], [q1, q2, p1, p2, y1, y2], (2, 4, 6))
r0, rr1, rr2 = sp.symbols('r0 rr1 rr2'); d = [q1, q2 - q1, -q2]; r = [r0, rr1, rr2]; dp = [p1, p2 - p1, -p2]
f = [dj + dj*rj for dj, rj in zip(d, r)]
report("FPU-beta, 2 masses, lift r_j=d_j^2 (7D)",
       [p1, p2, -(f[0] - f[1]), -(f[1] - f[2]), 2*d[0]*dp[0], 2*d[1]*dp[1], 2*d[2]*dp[2]], [q1, q2, p1, p2, r0, rr1, rr2], (2, 4, 6))
