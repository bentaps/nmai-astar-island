"""Theorem 6.5 (radial dynamics of the lifted isotropic Kahan map), verified by computer algebra and numerically.
1. With A = xi.eta = y - g, B = eta.eta' = y' - g, C = |eta|^2 = (Y B + Y' A)/(4 - Y - Y'),
   |xi|^2 = (A^2 + h^2 Y^2 m2)/C, the equation E_h(z) = e is equivalent to D(z) Q(y, y') = 0 (symbolic in all
   parameters), where Q is the symmetric biquadratic of the theorem.
2. The level-set quartic: |q|^2 = A(y) and b^2 = Q4(y) on {c_h = g, |M_h|^2 = m2, E_h = e}.
3. Continuous limit: Q(x, x + h w)/h^2 -> -4 w^2 + 16 [(x - g)(2e - u2 x^2) - m2].
4. Numerics: along an orbit (n = 3, u1 != 0), Q(y_k, y_{k+1}) = 0, b_k^2 = Q4(y_k), and (y_k, b_k) are samples at
   x0 + k tau of the solution of y'' = Q4'(y)/2, y'^2 = Q4(y).
Works with the shifted variable y~ = y + u1/u2 and g = c_h + u1/u2 (so u1 = 0 below)."""
import sympy as sp, numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
y, yn, h, u2, g, m2, e, x, w = sp.symbols('y yn h u2 g m2 e x w')
kap = h**2*u2; Y, Yn = 1 + kap*y/2, 1 + kap*yn/2
def Qbq(X, Xn):
    s, p = X + Xn, X*Xn
    P = 4*(kap*g - 1) + kap*h**2*(2*e*g*kap - 4*e + kap*m2)
    return P*s**2 - 4*kap*(e*kap*h**2 + 2)*s*p + 16*(e*kap*h**2 + 1)*p + 16*e*h**2*s - 16*h**2*(2*e*g + m2)
A, Bv = y - g, yn - g
C = (Y*Bv + Yn*A)/(4 - Y - Yn)
X = (A**2 + h**2*Y**2*m2)/C
P2 = (C - 2*A + X)/h**2; Q2 = (X + 2*A + C)/4
K = P2/2 + u2*y**2/2; D = 1 + kap*Q2 - h**4*u2/2*K
D_level = (2 - Y)*Y + kap*(y - g)
num = sp.numer(sp.together(K + kap*m2/2 - e*D))
print("[1] numerator of E_h(z) - e  =  const * D(z) * Q(y, y'):",
      sp.simplify(sp.cancel(num/(D_level*sp.expand(Qbq(y, yn))))).free_symbols <= {h, u2})
# 2. level-set quartic
a = sp.Symbol('a'); c_expr = 4/h**2*(g - x + a)
Ksol = c_expr/2 + u2*x**2/2
Aform = (1 + e*kap*h**2/2)*(x - g - kap*x**2/4) + e*h**2/2 - kap*h**2*m2/4
Eq = sp.expand(Ksol + kap*m2/2 - e*(1 + kap*a - h**4*u2/2*Ksol))
print("[2] E_h = e is linear in a = |q|^2 with root A(x):", sp.simplify(Eq.subs(a, Aform)) == 0)
Q4 = sp.expand(Aform*c_expr.subs(a, Aform) - m2*(1 + kap*x/2)**2)
print("    Q4 has degree", sp.degree(Q4, x), "in y")
# 3. continuous limit
lim = sp.limit(sp.expand(Qbq(x, x + h*w)/h**2), h, 0)
print("[3] continuous limit correct:", sp.expand(lim - (-4*w**2 + 16*((x - g)*(2*e - u2*x**2) - m2))) == 0)
# 4. numerics
rng = np.random.default_rng(4); n = 3; hv = 0.4; u2v = 0.8; u1v = 0.6; kv = hv*hv*u2v; Up = lambda yy: u2v*yy + u1v
def step(q, p, yy):
    al = hv*Up(yy); Yv = 1 + hv*al/2; eta = q + hv/2*p
    ynv = (yy*Yv + hv*(2*p - (al + hv*u1v)*q)@eta)/(Yv + kv*(q@eta)); aln = hv*Up(ynv)
    return ((1 - hv*aln/2)*q + hv*p)/Yv, ((1 - hv*al/2)*p - (al + aln)*q)/Yv, ynv
q, p, yy = rng.normal(size=n), rng.normal(size=n), 0.0; yy = q@q - hv*hv/4*p@p + 0.1
yt = yy + u1v/u2v; Yv = 1 + kv*yt/2; M2 = (q@q)*(p@p) - (q@p)**2; m2v = M2/Yv**2
Kv = p@p/2 + u2v*yt**2/2; Dv = 1 + kv*(q@q) - hv**4*u2v/2*Kv; ev = (Kv + kv*m2v/2)/Dv; gv = yy - q@q + hv*hv/4*p@p + u1v/u2v
vals = {h: hv, u2: u2v, g: gv, m2: m2v, e: ev}
Qf = sp.lambdify((y, yn), Qbq(y, yn).subs(vals)); Q4f = sp.lambdify(x, Q4.subs(vals)); dQ4f = sp.lambdify(x, sp.diff(Q4, x).subs(vals))
orb = [(q, p, yy)]
for k in range(60): orb.append(step(*orb[-1]))
Yk = np.array([o[2] + u1v/u2v for o in orb]); bk = np.array([o[0]@o[1] for o in orb])
print(f"[4] max |Q(y_k, y_k+1)| = {max(abs(Qf(Yk[k], Yk[k+1])) for k in range(60)):.1e},"
      f" max |b_k^2 - Q4(y_k)| = {max(abs(bk[k]**2 - Q4f(Yk[k])) for k in range(61)):.1e}")
sol = solve_ivp(lambda s, z: [z[1], dQ4f(z[0])/2], [0, 70], [Yk[0], bk[0]], rtol=1e-12, atol=1e-13, dense_output=True)
ts = np.linspace(0, 2, 20001); d = np.abs(sol.sol(ts)[0] - Yk[1]) + np.abs(sol.sol(ts)[1] - bk[1]); j = np.argmin(d[1:]) + 1
f = lambda T: np.sum((sol.sol(T) - np.array([Yk[60], bk[60]]))**2)
tau1 = minimize_scalar(lambda T: np.sum((sol.sol(T) - np.array([Yk[1], bk[1]]))**2), bounds=(ts[j-1], ts[j+1]), method='bounded', options={'xatol': 1e-13}).x
tau = minimize_scalar(f, bounds=(60*tau1 - 1e-3, 60*tau1 + 1e-3), method='bounded', options={'xatol': 1e-13}).x/60
err = max(np.abs(sol.sol(k*tau) - np.array([Yk[k], bk[k]])).max() for k in range(61))
print(f"    tau = {tau:.10f} (about 2h = {2*hv}); max |(y(k tau), y'(k tau)) - (y_k, b_k)|, k <= 60: {err:.1e}")
