# Scripts behind the note

Requirements: Python 3 with `numpy`, `scipy`, `sympy` and `matplotlib`. Run every script from
inside this folder (`cd code && python3 <script>.py`). Some scripts write `.npz` result files
here, and the figure scripts write PNGs to `../figures/`.

| Claim in the note | Script |
|---|---|
| Kahan conserves $y-q^2+\tfrac{h^2}4p^2$ for the lift of any quartic 1-DOF potential (symbolic) | `sym_kahan_check.py` |
| Modified energy $(\tfrac12p^2+\tfrac14y^2)/(1+\tfrac12h^2q^2)$ for the quartic oscillator (symbolic) | `sym_quartic_energy.py` |
| Modified constraint, modified energy and the measure $dq\,dp\,dy/(1+\tfrac12h^2q^2)$ (exact rational arithmetic) | `exact_measure.py`, `measure_search.py` |
| Orbits lie on two quadrics; identification of the invariants from orbit data | `quartic_kahan.py`, `quartic_kahan2.py`, `quartic_kahan3.py` |
| Quartic oscillator comparison table and Figure 1/2 data (KL, MQAV, SV, IEQ, STAG) | `quartic_compare.py`, `make_figs_quartic.py` |
| Convergence orders | `quartic_order.py` |
| Non-symmetric quartic potential (exact constraint, bounded energy, not integrable) | `quartic_general.py` |
| Lemma test on a random 7-dimensional system with a three-variable constraint | `lemma_random2.py` |
| Pendulum trigonometric lift (integrable) and phase drift with a redundant angle | `pendulum.py` |
| Failure on the paper's non-separable example; Kahan + reset drift | `nonseparable.py` |
| Stability test from the paper's Figure 2 (MQAV with Newton, staggered, explicit splitting) | `nonsep_methods.py`, `stag_probe.py` |
| Reduced quadratic-observable lift $(q^2,p^2,qp)$ | `reduced_lift.py` |
| Yoshida composition of the Kahan-lift map (fourth order, bounded errors) | `kahan_composition.py` |
| φ⁴ Klein–Gordon: methods, long-time series, accuracy, Figure 3 | `kg_pde.py`, `kg_series.py`, `make_figs_pde.py` |
| Kahan-lift predictor plus one Newton step of MQAV | `kl_predictor.py` |
| Allen–Cahn with and without constraint stabilisation versus IEQ, Figure 4 | `allen_cahn.py`, `make_fig_ac.py` |
| Sine-Gordon with two trigonometric lifts | `sine_gordon.py`, `sine_gordon_B.py` |
| FPU-β chain (generic sparse version, fast banded version, long runs) | `fpu.py`, `fpu_fast.py`, `fpu_long.py` |

Shared helpers: `qlift.py` (Kahan step, midpoint step, orbit-based algebraic-relation test) and
`vizstyle.py` (chart style).

Run times: most scripts finish in seconds to a few minutes. `kg_series.py`, `fpu.py` and
`fpu_long.py` take several minutes, and the symbolic scripts can take a few minutes.
