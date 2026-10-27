# Scripts behind the applications note

Requirements: Python 3 with `numpy`, `scipy` and `sympy`. Run every script from inside this folder
(`cd code-applications && python3 <script>.py`). `qlift.py` and `protected_span.py` are copies of the helpers in
`../code-invariants/`.

| Claim in [the note](../applications-and-paper-plan.md) | Script | Run time |
|---|---|---|
| Exact search, modulo a prime, for polynomial invariant densities $dz/P$ of Kahan maps (tool) | `darboux_modp.py` | — |
| Degree growth of iterates (algebraic entropy), exact modulo a prime (tool) | `degree_growth.py` | — |
| New integrable map: the lifted isotropic quartic oscillator conserves $c_h$, $M_h$, $E_h$ and the measure, numerically for $h=0.3,0.5,0.9$; modified angular momenta in $\mathbb R^3$ | `isotropic_quartic.py` | 1 min |
| The same integrals and measure are exact with $h$ symbolic | `isotropic_exact.py` | 2 min |
| Degree growth: quartic oscillator (integrable), non-symmetric quartic, isotropic quartic (integrable), open Toda | `integrability_degrees.py` | 15 min |
| Degree growth of isotropic variants: harmonic term, $\mathbb R^3$, anisotropic control | `isotropic_variants_degrees.py` | 10–40 min |
| Measures of multi-DOF lifted maps: isotropic quartic, Toda, random 3D Nambu systems have densities; $\varphi^4$, FPU, discrete NLS lifts do not (up to the stated degree) | `multidof_measures.py` | 6 min |
| Kahan–Toda: exact Liouville measure for $n=3,4,5$, not Poisson | `toda_measure.py` | 1 min |
| Polarisation does not make constraints protected (and the lift blows up) | `polarization_nogo.py` | seconds |
| The internal charge of a coupled $O(2)$ lattice is not protected | `o2_lattice_charge.py` | seconds |
| Stiff rotor chain: modified energy and circles conserved; stability for $h\omega\lesssim2$, blow-up beyond | `rotor_chain.py`, `rotor_chain_scan.py`, `rotor_chain_long.py` | 1–5 min |

Notes:

* The searches modulo a prime use a random step size $h$ (a random residue), so a density found there exists for
  generic $h$, and "none up to degree $d$" is an exact negative result for that degree (barring an unlucky prime).
* `degree_growth.py` restricts the $k$-th iterate to a random line and finds the degree of the resulting rational
  function from the nullity of a Padé-type linear system. Quadratic growth means integrable; exponential growth
  means not.
