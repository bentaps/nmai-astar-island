# Scripts behind the Kahan-invariants note

Requirements: Python 3 with `numpy`, `scipy` and `sympy`. Run every script from inside this folder
(`cd code-invariants && python3 <script>.py`). Most scripts finish in seconds. `verify_protected.py`,
`hierarchy_tests.py`, `phi6_reset.py` and the exact `sympy` searches take up to a few minutes.

| Claim in [the note](../kahan-invariants.md) | Script |
|---|---|
| Automatic protection test: space of quadratic integrals, lemma members, two-variable members (pencil cubic, resultant for three integrals, heuristic pencils beyond), protected span | `protected_span.py` |
| Protection table for ten lifted systems (§7) | `run_protected.py` |
| Block theorem: Hermitian forms, resonant (non-diagonalisable) block with squared denominator, nilpotent block; the resonant invariant is missed by mechanisms A and B (§1.3) | `block_theorem_checks.py` |
| Generalised Suslov system in $\mathbb R^4$: every normal-mode action is conserved in modified form (§1.3) | `suslov4d.py` |
| One-DOF family $U(\varphi)+\kappa$: exact one-step check of the pencil members for real, complex and mixed roots (§3) | `class_1dof_direct.py` |
| Mixed example $y=qp$: orbits on two quadrics, identification of the invariants, exact check, long run (§3) | `mixed_block.py`, `mixed_identify.py`, `mixed_exact.py` |
| Coupled pendula with a constant mass matrix, BH dimer, octic chain (§2.3, §5) | `verify_protected.py` |
| Triangular hierarchies: octic chain robustness, $\varphi^6$ Klein–Gordon lattice, resetting (§2.3) | `hierarchy_tests.py`, `phi6_reset.py` |
| Open Toda modified energy, back-substituted scheme equals lifted Kahan, Morse oscillator (§4, §5) | `toda_and_backsub.py` |
| Product-of-denominators measures (§6) | `measure_check.py`, `measure_product.py` |
| Non-symmetric quartic: exact degree-4 density at $h=\tfrac15$, and the closed form for all $h$ checked with $h$ symbolic (§6) | `measure_exact.py`, `measure_nonsym_allh.py` |
| Degree-4 densities for lifted quartic potentials and for random Nambu fields with a two-variable constraint (§6 conjecture) | `measure_nambu_search.py` |

Shared helpers: `qlift.py` (Kahan step, midpoint step, orbit-based algebraic-relation test).

Note on `protected_span.py`: with four or more independent quadratic integrals the two-variable search only
tries pencils between pairs of basis integrals, so it can miss members (it misses the FPU constraints, which the
lemma test catches). With up to three integrals the search is exact.
