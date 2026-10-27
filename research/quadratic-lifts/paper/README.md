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
| Theorem 6.1: the hand proofs are in the paper. Independent check: the closed-form step is Kahan's map; $c_h$, $M_h$, $E_h$ and the measure for $n=2,3$; functional independence. Symbolic $h,u_1,u_2$; output saved in `proofs/isotropic_proof_output.txt` | `proofs/isotropic_proof.py` | 8 min |
| Theorem 6.1: every displayed step of the hand proofs ($n=3$, symbolic $h,u_1,u_2$); output in `proofs/isotropic_hand_proof_check_output.txt` | `proofs/isotropic_hand_proof_check.py` | 1–2 min |
| Theorem 6.1, second cross-check with the step from the adjugate formula ($n=2$) | `proofs/isotropic_proof_adjugate.py` | 7 min (energy); the measure is slow with symbolic $u_1,u_2$, so pass `1/2 0` for 10 min |
| Remark 6.4: closed-form $O(n)$ step, $q_k\wedge q_{k+1}=hM_h$, central recurrence | `code/isotropic_explicit_step.py` | seconds |
| Figure 2 and the numbers in Section 8 (output saved in `fig_isotropic_numbers.txt`) | `code/fig_isotropic_numerics.py` | 1–2 min |
| Figure 1 (degree sequences; computed by the scripts listed in the docstring) | `code/fig_degrees.py` | seconds |
| Table 1, Sections 3–5 and 7 | scripts in [`../code-invariants/`](../code-invariants/), [`../code-applications/`](../code-applications/), [`../code-theory/`](../code-theory/); each folder's README maps claims to scripts | seconds to minutes |

## Open items before submission

* Author list, affiliations, acknowledgements, AI-use statement (TODO comments in `main.tex`).
* Reference [LP] (the Lie–Poisson manuscript): decide whether to merge, cite as companion paper, or wait for
  it. Corollary 5.4 depends on its Theorem 9(iii)(b), which is itself computer-assisted and based on
  unpublished computations.
* Literature check for integrable discretisations of isotropic anharmonic oscillators (Feng–Maruno 2008,
  discrete Newtonian systems in Suris 2003) and a conjugacy check of the isotropic map against them.
* Repository URL in the data-availability statement.
