"""Pendulum q'' = -sin q.  Trig lift z = (s, c, p) with s = sin q, c = cos q:
     s' = c p,  c' = -s p,  p' = -s      (quadratic)
Invariants: E = p^2/2 - c  (two variables p,c; A = s),  K = (s^2+c^2)/2 (two variables s,c; A = p).
CMOQ two-variable theorem predicts Kahan preserves
     E_h = p^2/2 - c - h^2 s^2/8,     K_h = (s^2+c^2) / (1 + h^2 p^2/4)."""
import numpy as np
from qlift import kahan_step, run, algebraic_relation_spectrum
F = lambda z: np.array([z[1]*z[2], -z[0]*z[2], -z[0]])
JF = lambda z: np.array([[0, z[2], z[1]], [-z[2], 0, -z[0]], [-1, 0, 0]])
h = 0.25; q0, p0 = 2.5, 0.0            # large-amplitude libration
for label, z0 in [("start on circle   ", np.array([np.sin(q0), np.cos(q0), p0])),
                  ("start on mod. leaf", np.array([np.sin(q0), np.cos(q0), p0])*np.r_[np.sqrt(1+h**2*p0**2/4)*np.ones(2), 1])]:
    Z = run(lambda z: kahan_step(F, JF, z, h), z0, 200000)
    s, c, p = Z.T
    Eh = p**2/2 - c - h**2*s**2/8; Kh = (s**2 + c**2)/(1 + h**2*p**2/4)
    q = np.arctan2(s, c); Etrue = p**2/2 - np.cos(q)
    print(f"{label}: drift E_h {np.max(abs(Eh-Eh[0])):.1e}, drift K_h {np.max(abs(Kh-Kh[0])):.1e}, "
          f"max|s^2+c^2-1| {np.max(abs(s**2+c**2-1)):.2e}, true-energy error (angle from atan2) {np.max(abs(Etrue-Etrue[0])):.2e}")
sv, _ = algebraic_relation_spectrum(Z[:3000], 2)
print("orbit lies on intersection of two quadrics? smallest sing. vals (deg 2):", sv[-3:])

# Variant with the angle kept as a separate variable: (q, p, s, c), q' = p.  Does atan2(s,c) - q drift?
F4 = lambda z: np.array([z[1], -z[2], z[3]*z[1], -z[2]*z[1]])
JF4 = lambda z: np.array([[0, 1, 0, 0], [0, 0, -1, 0], [0, z[3], 0, z[1]], [0, -z[2], -z[1], 0]])
for q0, p0 in [(2.5, 0.0), (0.0, 2.5)]:   # libration and rotation
    z0 = np.array([q0, p0, np.sin(q0), np.cos(q0)])
    Z = run(lambda z: kahan_step(F4, JF4, z, h), z0, 200000)
    ph = np.unwrap(np.arctan2(Z[:, 2], Z[:, 3])) - Z[:, 0]
    ph -= np.round(ph[0]/(2*np.pi))*2*np.pi
    n = len(ph)
    print(f"(q,p,s,c) lift, q0={q0}, p0={p0}: phase mismatch atan2(s,c)-q: first 1%: {np.max(abs(ph[:n//100])):.2e}, last 1%: {np.max(abs(ph[-n//100:])):.2e}")
