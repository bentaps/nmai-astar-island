# The Lie–Poisson draft, and whether (and why) to publish

*Answers four questions about the draft "Kahan's method for Lie–Poisson systems: orbit hulls,
measures and integrability" (22 October 2026). Is there theory in it that helps us? Is our work
enough to publish? When is an integrable map publishable? Why care, and is there value beyond
theory?*

> **Main takeaway.** The draft and our notes are two halves of one paper:
>
> * The draft supplies the general machinery (the Cayley form and midpoint theorem) and **proves
>   our 3D Nambu measure conjecture**.
> * We supply what the draft lacks: quadratic lifts, a block theorem that also works when the
>   matrices do *not* commute, and a **new integrable family in every dimension** for a *special*
>   (rotation-invariant) Hamiltonian, exactly the kind of case the draft lists as unexplained.
>
> Together that is publishable. On its own, our material needs proofs for the isotropic family
> and a literature check.

## 1. What the draft gives us

**Notation.** As in the draft: $t=h/2$; a quadratic field is written $f(x)=A(x)x$ with $A$ affine
(after adding a constant coordinate $z=1$), so that Kahan's map is
$(I-tA(x))\,x'=(I+tA(x'))\,x$ (its Theorem 1, the "Cayley form").

| Draft result | What it does for our work |
|---|---|
| **Theorem 2 (midpoint theorem):** if all $A(y)$ commute, then $x$ and $x'$ share a midpoint $u=(I-tA(x))^{-1}x$, every quadratic form $q$ for which the $A(y)$ are skew gives the exact integral $q(u)$, and $dx/\rho$ is preserved when $R_u$ has a spectrum symmetric under $\lambda\mapsto-\lambda$ | It **contains** our companion lemma, our single-speed block theorem (including the resonant block with its squared denominator) and CMOQ's two-variable theorem. It also yields our **exponential/harmonic Liouville-measure theorem**, once the force is placed in the $z$-column (see below). Then the $A(y)$ commute (checked symbolically) and $R_u$ is block anti-diagonal, so its spectrum is symmetric. Our three-shear proof is an alternative proof. |
| — | It does **not** contain our **several-speeds block theorem with non-commuting fields**. In the isotropic lift the natural $A(y)$ do not commute ($[A(y_1),A(y_2)]=\operatorname{diag}(y_1-y_2,\dots)$), yet the second-derivative operators act as scalars on the angular momenta. That is exactly what produces $M_h=M/(1+\tfrac12h^2U'(y))$. This is our main theoretical addition. |
| **Theorem 9(iii)(b):** a 3D hull system that is a Nambu system $\nabla C\times\nabla H$ with quadratic $C$, $H$ and unimodular core has a polynomial density of degree $\le4$ (computer-assisted proof) | Every 3D Nambu system with quadratic $C$ has a unimodular (Bianchi class A) core, and linear terms in $C$ are the draft's constant 2-cocycle. So this **proves our Nambu measure conjecture**. Corollary for us: every 1-DOF lift with one auxiliary variable, i.e. every quartic potential, gives a Kahan map that preserves a measure. The back-substituted method is therefore **area-preserving on the shifted sheet**. |
| **Orbit hulls, Remark 13, Conjecture 12:** for *generic* quadratic $H$, structure survives only for 2-step nilpotent algebras or one degree of freedom | Our lifted systems are affine Lie–Poisson systems on nilpotent algebras. The isotropic lift lives on a 3-step nilpotent algebra with $r=n$ and $\delta=1$. So for $n\ge2$ and generic $H$ the draft predicts no measure, matching our negative results for $\varphi^4$, FPU and discrete NLS. Our isotropic family is a **special-$H$ exception with a linear symmetry**, in the same category as the draft's unexplained cases (Lagrange, Kirchhoff, Clebsch, Manakov). |
| Table 2 lists special integrable cases it cannot explain | Our "linear symmetry plus one degree of freedom" principle plausibly explains the **Lagrange top and Kirchhoff case** (both axially symmetric: a linear integral plus a 1-DOF reduction). It does **not** explain Clebsch or Manakov on $\mathfrak{so}(4)$. Those survive with hidden quadratic integrals, so our earlier statement that hidden symmetries are always destroyed is **false in general**. It holds in every *lifted* system we tested; the previous note is corrected accordingly. |

**The factorisation for exponential potentials.** For Toda, on the variables $(p,u,z)$, write
$\dot p=(Lu)\,z$ and $\dot u=\operatorname{diag}(Dp)\,u$. So $A(x)$ has the column $Lu$ in the
$p$-rows and $z$-column, the block $\operatorname{diag}(Dp)$ on $u$, and zeros elsewhere. Products
$A(y)A(y')=\operatorname{diag}(0,\operatorname{diag}(Dp)\operatorname{diag}(Dp'),0)$ are symmetric in
$y,y'$, so the $A(y)$ commute. Harmonic variables work the same way, with a $2\times2$ rotation block
per angle. Separately: the degree sequences of the draft's "measure + Casimir" rows,
$3,9,23,57,139,337$, are the same as ours for the non-symmetric quartic and three-particle Toda.

## 2. Is the current work enough to publish?

**Yes, for one solid paper, once four things are done.** The cleanest route is to **merge with the
draft**, or to write a companion paper that cites it.

What is already strong:

1. **A new integrable family:** lifted isotropic quartic oscillators, in every dimension, with
    closed-form integrals ($c_h$, $M_{ab,h}$, $E_h$) and measure, and quadratic degree growth.
    It is a natural discretisation of a classical system, it is not covered by the draft's
    generic theory, and it comes from a new route (lift, then Kahan).
2. **The several-speeds block theorem with non-commuting fields:** a genuine extension of the
    midpoint theorem.
3. **The lift theory:**
    * which constraints are protected (squares, affine factors, opposite scaling with partners);
    * no-go results (polarisation, the one-level rule, prolongation);
    * back-substitution into one-step methods on the original variables.
4. **Area preservation for all quartic 1-DOF lifts:** now a theorem, via the draft's Theorem
    9(iii)(b).
5. **Systematic evidence:** exact measure searches and degree growth across about 30 systems.

What is missing before submission:

1. **Proofs for the isotropic family.** Plane preservation reduces every $n$ to the 5D map, so
    the identities for $c_h$, $M_h$, $E_h$ and the measure can be proved exactly by computer
    algebra, plus an integrability argument: three integrals, a measure and an $SO(2)$ symmetry in
    5D give quasi-periodic motion on 2-tori. Currently they are exact only at random points.
2. **A literature check** for integrable discretisations of central-force or isotropic
    anharmonic problems, and for Kahan–Toda's measure (now known to follow from the draft).
3. **A decision on the magnetic case:** report it as an observation (an integral exists, its
    mechanism is open), or explain it.
4. **Choice of audience.** For J. Phys. A or Nonlinearity, lead with the integrable family and
    the theorems. For BIT or IMA J. Numer. Anal., lead with linearly implicit integrators, with
    the competitor comparison (polarised two-step schemes, SAV/IEQ, midpoint).

## 3. When is an integrable map interesting enough to publish?

Referees in discrete integrable systems typically ask:

1. **Is it new up to conjugacy?** It must not be a known map (QRT maps, discrete Painlevé
    autonomous limits, known Kahan–Hirota–Kimura maps) in disguise. Invariants such as degree
    sequences, singularity patterns and the geometry of the invariant curves help decide this.
2. **Is integrability proved, and is the structure explicit?** Explicit integrals, an invariant
    measure or Poisson structure, ideally a Lax pair or an explanation. "Degree growth is
    quadratic" is evidence, not a proof.
3. **Is it a family or a mechanism?** One map is a curiosity. A mechanism producing many maps
    (CMOQ's two-variable theorem; QRT) or a family in arbitrary dimension is a contribution.
    Higher dimensions matter: planar integrable maps are abundant, integrable maps in $\mathbb R^n$
    for every $n$ are rare.
4. **Does it come from somewhere natural?** An integrability-preserving discretisation of a
    classical system (Euler top, Lagrange top, Toda, central forces) is valued in its own right
    (Suris's "problem of integrable discretisation").
5. **Does it connect to other structures?** Elliptic fibrations, explicit solutions,
    de-autonomisation to discrete Painlevé equations, Poisson geometry.

**Our isotropic family scores well on 3 and 4,** reasonably on 2 once the proofs are written, and
needs work on 1 (a conjugacy check against known 4D integrable maps) and 5 (an explicit solution in
elliptic functions would be a strong addition).

## 4. Why do we care about new integrable maps?

* **They are the exactly solvable models of discrete time.** Discrete integrable systems sit
    underneath discrete differential geometry, discrete Painlevé equations and orthogonal
    polynomials, cluster algebras, and integrable lattice models. New examples test and refine
    the general theory, such as the draft's Conjecture 12.
* **They show what a numerical method can preserve.** An integrable discretisation reproduces the
    qualitative long-time dynamics exactly: invariant tori, no artificial chaos, no drift. This
    is the strongest possible form of "geometric integration". Kahan's method is unusual in
    producing such maps while costing one linear solve per step.
* **They make good benchmarks and building blocks.** Exactly integrable maps are reference
    solutions, and integrable pieces can be composed. For example, splitting methods for
    near-integrable problems use an integrable discretisation of the main part, as Wisdom–Holman
    maps do for Kepler.

## 5. Is there value beyond a theoretically interesting method?

Yes, but in specific places:

* **Linearly implicit structure-preserving integrators.**
    * Per step: one linear solve instead of a Newton iteration, with exactly conserved modified
        invariants.
    * Stability: well beyond explicit stability limits when the stiff terms sit in the linear part
        of the lift (Klein–Gordon/$\varphi^4$ lattices; the stiff isotropic oscillator at
        $h\omega=5$ with a quarter of the midpoint rule's solves).
    * Constraints never drift, so the lifted method is a genuine one-step method on the
        original variables.

    That is the same practical niche as the popular SAV/IEQ schemes, with stronger invariant
    properties.
* **Exact volume preservation plus reversibility** for exponential and trigonometric potentials
    (Toda chains, rotor and XY chains, coupled pendula), with a modified energy. Volume
    preservation and reversibility are exactly what Hybrid Monte Carlo needs, so lifted Kahan is
    a candidate linearly implicit HMC integrator for such potentials. This is untested; the
    acceptance rates would have to be checked.
* **Limits to state honestly:**
    * no modified energy and no measure for multi-DOF quartic lattices;
    * step-size limits when stiffness enters through lifted nonlinear terms;
    * polarised two-step schemes conserve a modified energy for cubic wave equations, where we
        do not.

## 6. Recommended next steps

1. Merge with (or cite) the draft. Present our block theorem as the non-commuting extension of
    its midpoint theorem, and our measure results as applications of its Theorems 2(c) and
    9(iii)(b).
2. Prove the isotropic family's integrals and measure exactly (5D computer algebra, by plane
    reduction), and state its integrability as a theorem.
3. Run a conjugacy and literature check for the isotropic maps; try an explicit solution.
4. Optional, for a numerics paper: an HMC experiment with a Toda-type or rotor potential, and a
    lattice comparison against polarised two-step schemes.
