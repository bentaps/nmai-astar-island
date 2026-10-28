"""Floating-point check of Proposition 7.2 (magnetic case), independent of the closed form: the Kahan step is
computed by a generic 5x5 linear solve, z' = z + h (I - h/2 F'(z))^{-1} F(z).  Checks along a 2000-step orbit the
conservation of c_h, L_h and Ecal_h, the discrete Larmor identity for the step and for its inverse, and
det DPhi = P^B(z')/P^B(z)."""
import numpy as np
rng = np.random.default_rng(3)
h, B, u2 = 0.3, 0.7, 1.1
kap, beta = h*h*u2, h*B/2
def step(q, v, y):
    # Kahan step by solving the 5x5 linear system z' = z + h (I - h/2 F')^{-1} F
    z = np.array([*q, *v, y])
    F = np.array([v[0], v[1], -2*u2*y*q[0] - B*v[1], -2*u2*y*q[1] + B*v[0], 2*q@v])
    J = np.array([[0,0,1,0,0],[0,0,0,1,0],[-2*u2*y,0,0,-B,-2*u2*q[0]],[0,-2*u2*y,B,0,-2*u2*q[1]],[2*v[0],2*v[1],2*q[0],2*q[1],0]])
    zn = z + h*np.linalg.solve(np.eye(5) - h/2*J, F)
    return zn[:2], zn[2:4], zn[4], J
def inv(q, v, y):
    a, c, M = q@q, v@v, q[0]*v[1] - q[1]*v[0]
    Y = 1 + kap*y/2; ch = y - a + h*h/4*c; D = (2 - Y)*Y + kap*(y - ch); K = c/2 + u2*y*y/2
    PB = Y*(D - beta**2) + 2*beta**2 + h*beta*kap*M
    PL = M*(D + beta**2) - B*(2*a*Y + (1 + Y)*ch + y/2*(Y*Y - 2*Y - 2 + beta**2))
    PE = (2 + kap*ch)*K*Y + kap/2*M*M*(4 - Y) + B*B/2*(y - (a + ch)*(2 - Y)) - kap*B*M*(y/2 + a + ch)
    return ch, PL/PB, PE/((2 + kap*ch)*PB), PB
q, v = rng.normal(size=2)*0.6, rng.normal(size=2)*0.6; y = q@q
I0 = inv(q, v, y)
L = lambda q, v: q[0]*v[1] - q[1]*v[0] - B/2*(q@q)
maxdev = 0; maxlarm = 0; maxmeas = 0
for k in range(2000):
    qn, vn, yn, J = step(q, v, y)
    Jn = step(qn, vn, yn)[3]
    # Larmor identities and the measure
    Y, Yn = 1 + kap*y/2, 1 + kap*yn/2
    w, s = q[0]*qn[1] - q[1]*qn[0], q@qn
    maxlarm = max(maxlarm, abs(Y*w - beta*s - h*L(q, v)), abs(Yn*w - beta*s - h*L(qn, vn)))
    detDPhi = np.linalg.det(np.eye(5) + h/2*Jn)/np.linalg.det(np.eye(5) - h/2*J)
    maxmeas = max(maxmeas, abs(inv(qn, vn, yn)[3]/inv(q, v, y)[3] - detDPhi))
    q, v, y = qn, vn, yn
    I = inv(q, v, y)
    maxdev = max(maxdev, *[abs(I[i] - I0[i]) for i in range(3)])
print("max drift of (c_h, L_h, Ecal_h) over 2000 steps:", maxdev)
print("max Larmor residual (forward and inverse):", maxlarm)
print("max measure residual:", maxmeas)
