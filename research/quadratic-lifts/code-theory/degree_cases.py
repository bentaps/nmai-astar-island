"""Algebraic-entropy (degree growth) tests for lifted Kahan maps of candidate integrable systems.
Usage: python3 degree_cases.py <case>   (cases listed in CASES). Quadratic growth = integrable, exponential = not."""
import sympy as sp, sys, time
from degree_growth import degree_sequence
R = sp.Rational

def quartic2(beta, gamma):
    """V = q1^4 + beta q1^2 q2^2 + gamma q2^4, diagonal lift y_k = q_k^2."""
    q1, q2, p1, p2, y1, y2 = V = sp.symbols('q1 q2 p1 p2 y1 y2')
    return [p1, p2, -q1*(4*y1 + 2*beta*y2), -q2*(2*beta*y1 + 4*gamma*y2), 2*q1*p1, 2*q2*p2], list(V)

def garnier(a1, a2):
    """H = |p|^2/2 + (a1 q1^2 + a2 q2^2)/2 + |q|^4/4, lift y = |q|^2."""
    q1, q2, p1, p2, y = V = sp.symbols('q1 q2 p1 p2 y')
    return [p1, p2, -(a1 + y)*q1, -(a2 + y)*q2, 2*(q1*p1 + q2*p2)], list(V)

def isotropic(n):
    q = sp.symbols(f'q1:{n+1}'); p = sp.symbols(f'p1:{n+1}'); y = sp.Symbol('y')
    return list(p) + [-y*qi for qi in q] + [2*sum(qi*pi for qi, pi in zip(q, p))], list(q) + list(p) + [y]

def sextic_sl2():
    """H = |p|^2/2 + |q|^6/6 in R^2, lift closed under sl(2): y=|q|^2, w=y^2, eta=q.p, zeta=|p|^2."""
    q1, q2, p1, p2, y, w, eta, zeta = V = sp.symbols('q1 q2 p1 p2 y w eta zeta')
    return [p1, p2, -w*q1, -w*q2, 2*eta, 4*y*eta, zeta - w*y, -2*w*eta], list(V)

def toda2():
    a1, a2, u = V = sp.symbols('a1 a2 u')
    return [-u, u, u*(a1 - a2)], list(V)

CASES = {
    "quartic 1:2:1 (isotropic, diagonal lift)": lambda: quartic2(2, 1),
    "quartic 1:0:1 (decoupled)": lambda: quartic2(0, 1),
    "quartic 1:6:1 (separable after rotation)": lambda: quartic2(6, 1),
    "quartic 1:6:8 (integrable, quartic integral)": lambda: quartic2(6, 8),
    "quartic 1:12:16 (integrable, quartic integral)": lambda: quartic2(12, 16),
    "quartic 1:3:2 (non-integrable control)": lambda: quartic2(3, 2),
    "Garnier a=(1,2)": lambda: garnier(1, 2),
    "isotropic quartic n=4 (9D)": lambda: isotropic(4),
    "sextic isotropic, sl(2)-closed lift (8D)": sextic_sl2,
    "open Toda n=2": toda2,
}
def quartic161_adapted():
    """V = q1^4 + 6 q1^2 q2^2 + q2^4 = (u^4 + v^4)/2 with u = q1 + q2, v = q1 - q2; lift Yu = u^2, Yv = v^2
    (adapted to the coordinates in which the system decouples)."""
    q1, q2, p1, p2, Yu, Yv = V = sp.symbols('q1 q2 p1 p2 Yu Yv')
    u_, v_ = q1 + q2, q1 - q2
    return [p1, p2, -2*u_*Yu - 2*v_*Yv, -2*u_*Yu + 2*v_*Yv, 2*u_*(p1 + p2), 2*v_*(p1 - p2)], list(V)

def magnetic_isotropic(om):
    """q' = v, v' = om J v - y q (uniform magnetic field + |q|^4/4), lift y = |q|^2."""
    q1, q2, v1, v2, y = V = sp.symbols('q1 q2 v1 v2 y')
    return [v1, v2, -om*v2 - y*q1, om*v1 - y*q2, 2*(q1*v1 + q2*v2)], list(V)

def two_rotors():
    """H = (p1^2 + p2^2)/2 - cos(q1 - q2), lift (s, c) of theta = q1 - q2."""
    p1, p2, s, c = V = sp.symbols('p1 p2 s c')
    return [-s, s, c*(p1 - p2), -s*(p1 - p2)], list(V)

CASES.update({
    "quartic 1:6:1, lift adapted to the rotated coordinates": quartic161_adapted,
    "isotropic quartic in a magnetic field (omega=1)": lambda: magnetic_isotropic(1),
    "two coupled rotors, -cos(q1 - q2)": two_rotors,
})

if __name__ == "__main__":
    key = [k for k in CASES if sys.argv[1] in k][0]
    F, v = CASES[key](); t = time.time()
    print(f"{key:48s} degrees {degree_sequence(F, v, K=6, mmax=400)} ({time.time()-t:.0f}s)", flush=True)
