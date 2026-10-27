"""Degree growth for variants of the lifted isotropic quartic: with a harmonic term, and in R^3 (7D map)."""
import sympy as sp, time
from degree_growth import degree_sequence
R = sp.Rational
q1, q2, q3, p1, p2, p3, Y = sp.symbols('q1 q2 q3 p1 p2 p3 Y')
# H = |p|^2/2 + |q|^2/2 + |q|^4/4 in R^2:  p' = -(1 + y) q
cases = [("isotropic, |q|^2/2 + |q|^4/4 in R^2 (5D)", [p1, p2, -(1 + Y)*q1, -(1 + Y)*q2, 2*(q1*p1 + q2*p2)], [q1, q2, p1, p2, Y]),
         ("isotropic, |q|^4/4 in R^3 (7D)", [p1, p2, p3, -Y*q1, -Y*q2, -Y*q3, 2*(q1*p1 + q2*p2 + q3*p3)], [q1, q2, q3, p1, p2, p3, Y]),
         ("anisotropic, (q1^2 + 2 q2^2)^2/4 in R^2 (5D)", [p1, p2, -Y*q1, -2*Y*q2, 2*(q1*p1 + 2*q2*p2)], [q1, q2, p1, p2, Y])]
for name, F, v in cases:
    t = time.time(); print(f"{name:50s} degrees {degree_sequence(F, v, K=7, mmax=900)} ({time.time()-t:.0f}s)", flush=True)
