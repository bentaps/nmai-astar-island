"""Section 8: how far beyond h*omega = 2 the lifted Kahan map stays bounded, as a function of the amplitude.
Stiff isotropic oscillator in R^3, H = |p|^2/2 + om^2 |q|^2/2 + |q|^4/4, om = 10 (u2 = 1/2, u1 = om^2/2).
For each amplitude A and step size, 20 random orbits with |q0| = A, |p0| = om*A (independent uniformly random
directions), y0 on the shifted sheet y0 = |q0|^2 - h^2 |p0|^2/4, integrated to T = 200 with the closed-form step of
Remark 6.4.  An orbit "escapes" if the relative energy error exceeds 100 (or the iterate is not finite), the same
criterion as in fig_stepsize_scan.py.  Also prints the sign of D(z0) (Proposition 6.3 needs D(z0) > 0) and, for
comparison, the number of escapes of the Strang splitting with the exact harmonic flow from the same initial data.
Writes ../stability_amplitude_numbers.txt.  Run time about a minute."""
import numpy as np
n, om = 3, 10.0; u2, u1 = 0.5, om**2/2; Up = lambda y: u2*y + u1
H = lambda q, p: p@p/2 + om**2*(q@q)/2 + (q@q)**2/4
T, CAP, NORB = 200.0, 1e2, 20
def unit(rng):
    v = rng.normal(size=n); return v/np.linalg.norm(v)
def escapes(q0, p0, h):
    q, p = q0.copy(), p0.copy(); y = q@q - h*h/4*p@p; kap = h*h*u2; E0 = H(q0, p0)
    for k in range(int(round(T/h))):
        al = h*Up(y); Y = 1 + h*al/2; eta = q + h/2*p
        yn = (y*Y + h*(2*p - (al + h*u1)*q)@eta)/(Y + kap*(q@eta)); aln = h*Up(yn)
        q, p, y = ((1 - h*aln/2)*q + h*p)/Y, ((1 - h*al/2)*p - (al + aln)*q)/Y, yn
        e = abs(H(q, p) - E0)/E0
        if not np.isfinite(e) or e > CAP: return True
    return False
def escapes_split(q0, p0, h):
    "Strang splitting flow(h/2)-kick(h)-flow(h/2) with the exact harmonic flow, for comparison"
    c, s = np.cos(om*h/2), np.sin(om*h/2); q, p = q0.copy(), p0.copy(); E0 = H(q0, p0)
    for k in range(int(round(T/h))):
        q, p = c*q + s/om*p, -om*s*q + c*p
        p = p - h*(q@q)*q
        q, p = c*q + s/om*p, -om*s*q + c*p
        e = abs(H(q, p) - E0)/E0
        if not np.isfinite(e) or e > CAP: return True
    return False
def D0(q, p, h):
    y = q@q - h*h/4*p@p; K = p@p/2 + u2/2*(y + u1/u2)**2; return 1 + h*h*u2*(q@q) - h**4*u2/2*K
lines = []
for A in (0.5, 1.0, 2.0, 3.0):
    row = []
    for hw in (1.5, 2.5, 5.0, 10.0):
        rng = np.random.default_rng(1); h = hw/om; cnt = 0; dpos = 0; spl = 0
        for i in range(NORB):
            q0, p0 = A*unit(rng), om*A*unit(rng)
            cnt += escapes(q0, p0, h); dpos += D0(q0, p0, h) > 0; spl += escapes_split(q0, p0, h)
        row.append(f"h*om={hw:4.1f}: {cnt:2d}/{NORB} escape (D(z0)>0 for {dpos}/{NORB}; splitting: {spl} escape)")
    lines.append(f"|q0| = {A}:  " + ";  ".join(row))
open("../stability_amplitude_numbers.txt", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
