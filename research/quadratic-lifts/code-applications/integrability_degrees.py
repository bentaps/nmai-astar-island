"""Algebraic entropy (degree growth of iterates, computed exactly mod a prime on a random line).
Quadratic growth = integrable; exponential growth = not integrable. Takes about 10-15 minutes in total."""
import sympy as sp, time
from degree_growth import degree_sequence
R = sp.Rational
q, p, y = sp.symbols('q p y'); q1, q2, p1, p2, Y = sp.symbols('q1 q2 p1 p2 Y'); a1, a2, a3, u1, u2 = sp.symbols('a1 a2 a3 u1 u2')
cases = [("quartic oscillator lift (3D; integrable, calibration)", [p, -q*y, 2*q*p], [q, p, y]),
         ("non-symmetric quartic lift (3D)", [p, -(y/3 + 2*q*(y/2 + q/3 + R(1, 2))), 2*q*p], [q, p, y]),
         ("isotropic quartic lift y=|q|^2 (5D)", [p1, p2, -q1*Y, -q2*Y, 2*(q1*p1 + q2*p2)], [q1, q2, p1, p2, Y]),
         ("open Toda n=3 (5D)", [-u1, u1 - u2, u2, u1*(a1 - a2), u2*(a2 - a3)], [a1, a2, a3, u1, u2])]
for name, F, v in cases:
    t = time.time(); print(f"{name:55s} degrees {degree_sequence(F, v, K=7, mmax=900)} ({time.time()-t:.0f}s)", flush=True)
