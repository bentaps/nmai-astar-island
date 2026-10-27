# Scripts behind the dimensions-and-integrable-class note

Requirements: Python 3 with `numpy`, `scipy` and `sympy`. Run from inside this folder. `qlift.py`,
`protected_span.py`, `darboux_modp.py`, `degree_growth.py` and `exact_density.py` are shared helpers (copies of the
ones in `../code-applications/` and `../code-invariants/`).

| Claim in [the note](../dimensions-and-integrable-class.md) | Script | Run time |
|---|---|---|
| Degree growth for 13 lifted systems (isotropic $n=4$, magnetic isotropic, decoupled, $1{:}2{:}1$, $1{:}6{:}1$ with diagonal and adapted lifts, $1{:}6{:}8$, $1{:}12{:}16$, $1{:}3{:}2$, Garnier, two-particle Toda, two rotors, sextic isotropic); run one case per call, e.g. `python3 degree_cases.py "1:6:8"`. The isotropic $n=2,3$ and harmonic-term rows of the note come from `../code-applications/integrability_degrees.py` and `../code-applications/isotropic_variants_degrees.py` | `degree_cases.py` | 2 s – 2 min per case |
| Isotropic family in $\mathbb R^n$: closed-form $E_h$ for every $n$ and quadratic $U$ | `isotropic_general.py` | 1 min |
| Isotropic family: invariant densities (exact search) and the closed-form measure for every $n$ | `isotropic_measure_nd.py` | 1 min |
| Plane preservation in $\mathbb R^4$; stiff isotropic oscillator: lifted Kahan versus implicit midpoint and Störmer–Verlet | `isotropic_stiff_numerics.py` | 10 s |
| Liouville-measure theorem for exponential and harmonic potentials (floating point, four systems) | `exp_trig_measure.py` | 10 s |
| The same theorem in exact rational arithmetic (mixed system, 3 DOF, non-diagonal mass matrix) | `exp_trig_measure_exact.py` | 2 min |
| Prolongation protects $v-f(x)$ but not the Euler top's integrals; the prolonged map is not integrable | `prolongation.py`, `prolongation_degrees.py` | 1 min |
| Magnetic isotropic oscillator: densities by exact search, and the rational integral $P_1/P_2$ | `magnetic_search.py`, `magnetic_integral.py` | 3 min, seconds |
