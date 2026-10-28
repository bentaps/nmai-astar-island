# How strong is the draft now, and what is left before submission?

> **Main takeaway.** After this round, every weakness on the earlier list has been addressed except
> coordinating with the companion manuscript [LP]. The draft is now a credible submission to *Nonlinearity*:
> expect "minor" or "major revision" rather than a rejection. It remains a safe fit for *J. Phys. A*.
>
> * **New results:**
>   * an explicit solution of the integrable family through a QRT map and elliptic functions;
>   * a closed-form measure for every quartic potential in one degree of freedom, so no result depends on [LP]
>     any more;
>   * proofs of integrability for every map that the survey flagged as integrable;
>   * the magnetic case, now explained with closed formulas;
>   * an honest step-size comparison against a splitting method.

## 0. Terms and notation used below

| Term | Meaning |
|---|---|
| ODE | ordinary differential equation, $\dot z=F(z)$ |
| DOF | degree of freedom: one position–momentum pair $(q,p)$ |
| $h$, $\Phi_h$ | the time step; one step of Kahan's method, $z\mapsto z'$ |
| Kahan's method | for a quadratic $F$: $\frac{z'-z}{h}=Q(z,z')+\frac12B(z+z')+b$. A step costs one linear solve. |
| quadratic lift | extra variables, e.g. $y=q^2$, that turn a non-quadratic ODE into a quadratic one in more variables |
| shifted sheet | the surface $y=q^2-\frac{h^2}{4}p^2$, which the lifted Kahan map never leaves. On it the method is an ordinary one-step method for $(q,p)$. |
| modified integral $I_h$ | a function $I+O(h^2)$ that Kahan's map conserves **exactly**, step after step |
| invariant measure (density) $\mathrm dz/P$ | a way of measuring volume that the map preserves exactly; $P$ is a polynomial |
| integrable map | a map with enough conserved quantities and a preserved measure that every orbit lies on a curve or a small torus; no chaos |
| degree growth $d_k$ | how fast the algebraic degree of the $k$-th iterate grows. Polynomial growth signals integrability, exponential growth signals chaos. |
| QRT map | Quispel–Roberts–Thompson map: the classic integrable map of the plane. It moves points along a family of curves that are each the zero set of one polynomial of degree two in each of two variables. |
| elliptic function | a doubly periodic function, e.g. Jacobi's $\mathrm{sn}$. "Sampled at equal steps" means $y_k=\mathfrak y(x_0+k\tau)$. |
| Nambu lift | the lift $\dot z=\nabla\tilde H\times\nabla c$ in $\mathbb R^3$ with $c=y-q^2$; for one-DOF quartic potentials it is unique |
| SV | the Störmer–Verlet method, the standard explicit integrator; unstable for $h\omega>2$ |
| impulse method | Strang splitting that solves the stiff linear part exactly and adds kicks from the nonlinear force |
| CRT | Chinese remainder theorem: combine results computed modulo several primes into one exact result |
| [LP] | the uploaded companion draft "Kahan's method for Lie–Poisson systems: orbit hulls, measures and integrability" |

## 1. What changed, weakness by weakness

| Earlier weakness | What the draft does now | Where |
|---|---|---|
| **Dependence on [LP].** Corollary 5.3 needed [LP]'s computer-assisted, unpublished Theorem 9(iii)(b). | **Removed.** A closed-form density $P$ for the Nambu lift of *every* potential $V=a_4q^4+a_3q^3+a_2q^2+a_1q$. It is a polynomial identity, checked exactly in under a second with all coefficients and $h$ symbolic. The area-form corollary is now unconditional. | Prop. 5.3, Cor. 5.4 |
| **Section 7 was empirical.** | **Every map in the integrable half of Table 1 is now proved integrable:** one DOF; the isotropic family and anything affinely conjugate to it (anisotropic mass matrix, the two-body problem); products of these; single exponentials and harmonics in any dimension; the magnetic case. Only the general principle (Conjecture 7.3) and the non-integrability of the other half remain evidence-based. Algebraic entropy is the accepted evidence for that. | Prop. 7.1 |
| **No explicit solution; few links to other structures.** | **The radial motion is a QRT map.** The lifted variable is an elliptic function sampled at equal steps. This gives the whole orbit explicitly and links the family to the classic theory of integrable maps of the plane. | Thm. 6.5, Cor. 6.6 |
| **The magnetic case was unexplained.** | **Closed formulas:** the step, three integrals, the measure, and a "discrete Larmor identity" linking the continuous angular momentum to consecutive positions. Verified exactly in 7 seconds and independently in floating point. | Prop. 7.2 |
| **Modest numerical gain.** | **An honest scan over 1961 step sizes** against the midpoint rule, SV and the impulse method (the natural explicit competitor). The lifted Kahan map has **one** failure band, at the step size the theory predicts. The impulse method has six resonance bands. The text also says where the lifted Kahan map loses: it is less accurate than the midpoint rule, and less accurate than the impulse method between its resonances. | Sec. 8, Fig. 3 |
| **Novelty of the integrable map (no conjugacy check).** | Partly addressed. The QRT reduction makes the structure explicit, and a conjugacy check against known maps is now a finite computation. The check itself has not been done. | open |

**An example of the new radial result.**
* Take $H=\frac12|p|^2+\frac12|q|^4$ in the plane, $h=0.35$.
* Plot each pair of consecutive values $(y_k,y_{k+1})$ of the lifted variable $y\approx|q|^2$. They lie exactly, to $10^{-14}$, on one curve $\mathcal Q(y,y')=0$ (Figure 2c).
* That curve is a slightly deformed copy of the continuous energy curve $\dot y^2=4[(y-\gamma)(2e-u_2y^2)-m^2]$.

**An example of the new measure.** For $V=\frac14q^4$ the density factorises as
$P=(1+\frac12h^2q^2)(3+\frac12h^2c_h-\frac14h^4\tilde H_h)$. The first factor is the density already known
for this case, and the second is a function of conserved quantities. For a lopsided potential such as
$V=\frac14q^4+\frac13q^3+\frac12q^2$ there is no conserved energy and the map is chaotic. It still preserves
area, by the same formula.

## 2. Scorecard against what referees ask of a paper on integrable maps

| Criterion | Before | Now | Why |
|---|---|---|---|
| Is the main map new? | medium | medium–strong | New as a Kahan map with closed-form integrals in every dimension, now with an explicit solution. No conjugacy check yet. |
| Is integrability proved, with explicit structure? | strong | strong | Integrals, measure, tori, and now the explicit solution. |
| Family or mechanism, not one example? | strong | strong | Block theorem, Liouville theorem, isotropic family, quartic density for all potentials. |
| Connections to other structures | weak–medium | medium–strong | QRT maps and elliptic curves (Thm. 6.5); still no Lax pair or Poisson structure. |
| Self-contained (no reliance on unpublished work) | weak | strong | [LP] is no longer used in any proof. |
| Survey section | weak (empirical) | medium–strong | All positive cases proved; one conjecture remains. |
| Numerics | medium | medium–strong | Fair comparison, including the method's weaknesses; one predicted failure band. |
| Depth of the main theorem | medium | medium | Proofs are still short. Two new results (Prop. 5.3, Prop. 7.2) are computer-algebra identities, not conceptual proofs. |

## 3. What a referee could still object to (most serious first)

1. **Overlap and priority with [LP].** No proof uses [LP] any more. The overlap remains: [LP]'s "midpoint
   theorem" covers the single-speed block theorem and the commuting measure results. If [LP] is yours, consider
   merging or submitting together; if not, coordinate with its authors.
2. **Two results are verified by computer algebra rather than proved by hand.** These are Prop. 5.3 and
   Prop. 7.2(b). Both are exact polynomial identities that run in seconds, and the scripts are in the
   repository, which is acceptable in this field. A referee may still ask *why* the formulas hold. Open problem 1
   (a block theorem without closure, or a discrete Larmor theorem) is the natural answer, and it is not yet
   found.
3. **Conjecture 7.3 remains a conjecture.** It is confirmed in every case tested, and Prop. 7.1 proves it in
   the cases where the symmetry reduction is a rotation or an affine splitting.
4. **The lifted Kahan map is not the most accurate integrator.** Its selling points are cost (an explicit
   $O(n)$ step), exact modified invariants, stability beyond the SV limit, and a predictable failure band.
5. **No conjugacy check** against Feng–Maruno (2008) or discrete Newtonian central-force maps.

## 4. Independent check of the new material

*To be completed from the independent reviewer's report (see below).*

## 5. Where to send it

* ***Nonlinearity***: now a realistic target. Its selling points are:
  * integrability in every dimension with an explicit elliptic solution;
  * a measure-preservation theorem for all quartic potentials, including the chaotic ones;
  * a proved list of which lifted systems stay integrable, plus a conjecture on why.
* ***J. Phys. A: Math. Theor.***: the safest home, since most Kahan papers appear there.
* ***Proc. R. Soc. A*** or ***Open Communications in Nonlinear Mathematical Physics***: also natural.

## 6. What would make it stronger still (best value first)

1. **Settle the [LP] relationship** (merge, or submit as companion papers).
2. **A conceptual proof of Prop. 7.2.** For example, a discrete Larmor theorem that derives the magnetic map
   from the non-magnetic one. Alternatively, a general "block theorem without closure".
3. **Extend Prop. 5.3** from quartic Nambu lifts to all three-dimensional quadratic Nambu systems with a
   closed-form density. This would give an independent proof of [LP]'s computer-assisted theorem.
4. **Conjugacy check** of the isotropic map against known integrable discretisations, using the QRT normal
   form.
5. **A Hybrid Monte Carlo test** of the volume-preserving Toda and rotor integrators.

## 7. Files

* Paper: `main.tex`, `main.pdf` (24 pages); `README.md` maps every result to the script that checks it.
* New proof scripts:
  * `proofs/quartic_density_proof.py`
  * `proofs/isotropic_radial_check.py`
  * `proofs/magnetic_proof.py`
  * `proofs/magnetic_float_check.py`
* New search and figure scripts:
  * `code/quartic_density_search.py`
  * `code/magnetic_darboux_search.py`
  * `code/fig_stepsize_scan.py`
* **GitHub:** pushing is still blocked. This session's GitHub access cannot create repositories and is not
  connected to the target repository. To finish:
  1. create an empty repository, e.g. `bentaps/kahan-quadratic-lifts`;
  2. give the Claude GitHub app access to it;
  3. tell me its name, and I will push the prepared branch.
