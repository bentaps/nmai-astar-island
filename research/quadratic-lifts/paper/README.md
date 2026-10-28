# Paper draft: Kahan's method on quadratic lifts

Draft for submission to *Nonlinearity*:
**"Kahan's method on quadratic lifts: modified invariants, Liouville measures and integrable maps in every
dimension"** ([`main.pdf`](main.pdf), source [`main.tex`](main.tex)).

The research notes this draft is built from are one folder up ([`../README.md`](../README.md),
[`../kahan-invariants.md`](../kahan-invariants.md), [`../applications-and-paper-plan.md`](../applications-and-paper-plan.md),
[`../dimensions-and-integrable-class.md`](../dimensions-and-integrable-class.md),
[`../lie-poisson-draft-and-publication.md`](../lie-poisson-draft-and-publication.md)).

## Build

```
latexmk -pdf main.tex
```

Plain `article` class. IOP journals accept any reasonable LaTeX at first submission; swap in the IOP class at
acceptance.

## What reproduces what

Requirements: Python 3 with `numpy`, `scipy`, `sympy`, `matplotlib`. Run the scripts from their own folder.

| Item in the paper | Script | Run time |
|---|---|---|
| Proposition 5.3 (closed-form invariant density for the Nambu lift of every quartic potential, hence Corollary 5.4 without [LP]): the measure identity as a polynomial identity in $(q,p,y,h,a_1,\dots,a_4)$; output in `proofs/quartic_density_proof_output.txt` | `proofs/quartic_density_proof.py` | < 1 s |
| Proposition 5.3: how the density was found (weighted-homogeneous ansatz, modular search, CRT) | `code/quartic_density_search.py` | 2 s |
| Theorem 6.1: the hand proofs are in the paper. Independent check: the closed-form step is Kahan's map; $c_h$, $M_h$, $E_h$ and the measure for $n=2,3$; functional independence. Symbolic $h,u_1,u_2$; output saved in `proofs/isotropic_proof_output.txt` | `proofs/isotropic_proof.py` | 8 min |
| Theorem 6.1: every displayed step of the hand proofs ($n=3$, symbolic $h,u_1,u_2$); output in `proofs/isotropic_hand_proof_check_output.txt` | `proofs/isotropic_hand_proof_check.py` | 1–2 min |
| Theorem 6.1, second cross-check with the step from the adjugate formula ($n=2$) | `proofs/isotropic_proof_adjugate.py` | 7 min (energy); the measure is slow with symbolic $u_1,u_2$, so pass `1/2 0` for 10 min |
| Remark 6.4: closed-form $O(n)$ step, $q_k\wedge q_{k+1}=hM_h$, central recurrence | `code/isotropic_explicit_step.py` | seconds |
| Theorem 6.5 and Corollary 6.6 (radial QRT map, explicit elliptic solution): numerator of $E_h-e$ is $\text{const}\cdot D\,\mathcal Q(y,y')$, the quartic $Q_4$, the limit $h\to0$, and a numerical check that $(y_k,b_k)$ samples the elliptic function at equal steps; output in `proofs/isotropic_radial_check_output.txt` | `proofs/isotropic_radial_check.py` | 2 s |
| Proposition 7.1 (proved integrable cases) | follows from the cited results; the degree sequences of Table 1 are a consistency check | — |
| Proposition 7.2 (magnetic case): closed-form step, discrete Larmor identity, $c_h$, $L_h$, $\mathcal E_h$, both determinants, measure, $B=0$ reductions, $h\to0$ limits, independence. Symbolic $h,B,u_2$; output in `proofs/magnetic_proof_output.txt` | `proofs/magnetic_proof.py` | 7 s |
| Proposition 7.2, floating-point cross-check with the step computed by a generic linear solve | `proofs/magnetic_float_check.py` | 1 s |
| Proposition 7.2: how the integrals were found (rotation-invariant Darboux search modulo primes, CRT and rational reconstruction) | `code/magnetic_darboux_search.py 1/5 1 1/2 0` | 1 s |
| Figure 2 and the numbers in Section 8 (output saved in `fig_isotropic_numbers.txt`) | `code/fig_isotropic_numerics.py` | seconds |
| Figure 3 (step-size scan: lifted Kahan, Strang splitting with the exact harmonic flow in both orders, midpoint, Störmer–Verlet; resonance bands; output saved in `fig_stepsize_numbers.txt`) | `code/fig_stepsize_scan.py` | 2 min |
| Section 8, stability against amplitude: escapes of 20 random orbits per amplitude and step size, lifted Kahan and splitting, with the sign of $D(z_0)$; output saved in `stability_amplitude_numbers.txt` | `code/stability_amplitude_scan.py` | 10 s |
| Figure 1 (degree sequences; computed by the scripts listed in the docstring) | `code/fig_degrees.py` | seconds |
| Table 1, Sections 3–5 and 7 | scripts in [`../code-invariants/`](../code-invariants/), [`../code-applications/`](../code-applications/), [`../code-theory/`](../code-theory/); each folder's README maps claims to scripts | seconds to minutes |

## Open items before submission

* Author list, affiliations, acknowledgements, AI-use statement (TODO comments in `main.tex`).
* Reference [LP] (the Lie–Poisson manuscript): decide whether to merge or cite it as a companion paper. No
  result of this paper depends on it any more: Proposition 5.3 gives the measure that Corollary 5.4 needs in
  closed form, so the earlier reliance on [LP] Theorem 9(iii)(b) is gone. The overlap described in the
  introduction (single-speed block theorem, commuting measure results) remains to be coordinated.
* Literature check for integrable discretisations of isotropic anharmonic oscillators (Feng–Maruno 2008,
  discrete Newtonian systems in Suris 2003). Theorem 6.5 reduces the radial part to a QRT map, so a conjugacy
  check against known QRT/discrete Newtonian maps is now a finite computation.
* Optional: a hand proof of Proposition 7.2(b) (currently computer algebra), ideally via a discrete Larmor
  theorem.
* Repository URL in the data-availability statement.
