"""Isotropic quartic in a uniform magnetic field (q' = v, v' = om J v - y q, y' = 2 q.v): find its invariant densities
exactly (rational h, rational om) and the integral given by their ratio."""
import sympy as sp
from exact_density import densities
q1, q2, v1, v2, y = V = sp.symbols('q1 q2 v1 v2 y'); om = sp.Integer(1); h = sp.Rational(1, 5)
F = [v1, v2, -om*v2 - y*q1, om*v1 - y*q2, 2*(q1*v1 + q2*v2)]
Ps = densities(F, list(V), 4, h)
for P in Ps: print("P =", P)
if len(Ps) >= 2: print("ratio P1/P2 =", sp.factor(Ps[0]/Ps[1]))
