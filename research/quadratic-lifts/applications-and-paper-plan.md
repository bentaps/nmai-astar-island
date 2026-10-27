# What the Kahan-lift invariants are good for, and what to write up

*Third note in this folder (after [the first note](README.md) and
[the Kahan-invariants note](kahan-invariants.md)). It answers four questions:*

1. *Can the ideas find genuinely new integrable maps, or new preserved measures and integrals?*
2. *What do they give for semi-discretised PDEs?*
3. *Can polarisation, $z_{ij}=x_i+x_j$, make every lift Kahan-friendly?*
4. *What should go into a paper?*

*Every claim is reproduced by a script in [`code-applications/`](code-applications/) (see §6).*

> **Main takeaway.** Yes, the ideas produce genuinely new results:
>
> * a new **integrable** 5-dimensional map (Kahan's method on the quadratic lift of the isotropic
>   quartic oscillator), with all integrals and its measure in closed form;
> * new **modified angular momenta**, which come from the block theorem with several
>   non-commuting speeds;
> * an **exact Liouville measure** for Kahan's method on Toda-type lattices;
> * evidence that **every** three-dimensional quadratic Nambu system has a Kahan-invariant
>   measure.
>
> For semi-discretised PDEs the gains are narrower. Cubic wave equations get a linearly
> implicit, symmetric one-step method with exactly conserved modified constraints, but no
> modified energy and, as I now show, no invariant measure. Only exponential and rotor lattices
> are fully protected, and even there protection does not mean stability: stiff terms must sit in
> the linear part of the lift. Your polarisation trick cannot widen the protected class: protection is
> decided by which linear combinations of the variables move affinely, and recombining variables
> linearly does not change that.

## Answers at a glance

| Question | Answer | Where |
|---|---|---|
| New integrable maps? | **Yes, a family.** Kahan on the lift $y=\lvert q\rvert^2$ of the isotropic quartic oscillator conserves three explicit integrals and a measure in $\mathbb R^2$, and its degree growth is quadratic (zero algebraic entropy). This also holds with a harmonic term and in $\mathbb R^3$. Most other lifted maps tested are not integrable, including Kahan–Toda with three particles. | §1 |
| New preserved measures or integrals? | **Yes.** Modified angular momenta $M_{ab}/(1+\tfrac12h^2U'(y))$ in any dimension. The exact Liouville measure $dp\,du/\prod u_k$ for Kahan–Toda (with a proof). Degree-4 densities for every 3D quadratic Nambu system (conjecture with exact evidence). | §2 |
| Semi-discretised PDEs? | Cubic wave equations ($\varphi^4$, FPU-$\beta$, nonlinear strings): every constraint protected, linearly implicit one-step method with no time-step restriction from the linear part, but no modified energy and no polynomial invariant measure. Toda and rotor lattices: energy protected too. Nonlinear Schrödinger, parabolic and coupled $O(n)$ lattices: not protected. | §3 |
| Polarisation $z_{ij}=x_i+x_j$? | **No.** The linear constraint is conserved, but the squares $y_{ii}-x_i^2$ cannot be exact integrals unless $\dot x_i$ is affine, and that is exactly the condition you started with. On your paper's cubic example the polarised lift blows up. | §4 |
| What to write? | A theory paper built around the block theorem, the protected span and back-substitution, with the new integrable family as the showcase, and the Nambu-measure conjecture as the main open theorem. | §5 |

---

## 0. Notation and terms

Notation as in the previous notes: $z$ is the lifted state, $\dot z=F(z)$ the quadratic lifted
field, $h$ the step, $z\mapsto z'$ one Kahan step, and a subscript $h$ marks the modified version
of an invariant that Kahan conserves exactly. New terms:

| Term | Meaning |
|---|---|
| degree growth, algebraic entropy | Kahan's map is birational: a rational map with a rational inverse. Restrict its $k$-th iterate to a random line $z=a+bt$; each coordinate becomes a rational function of $t$ of some degree $d_k$. **Integrable maps have polynomially growing $d_k$; chaotic ones grow exponentially** (algebraic entropy $\lim\frac1k\log d_k>0$). This is the standard integrability test for birational maps. |
| Darboux polynomial, cofactor | a polynomial $P$ with $P(z')=\lambda(z)P(z)$; $\lambda$ is its cofactor. Two Darboux polynomials with the same cofactor give an integral $P_1/P_2$. If $\lambda=\det D\Phi$, then $dz/P$ is a preserved measure. |
| Nambu system (3D) | $\dot z=\nabla H\times\nabla C$; both $H$ and $C$ are integrals. Every 1-DOF lift with one auxiliary variable is one, with $C=y-\varphi(x)$. |
| isotropic | rotation-invariant, e.g. a potential depending only on $\lvert q\rvert^2$ |

All searches are **exact**: linear algebra modulo a large prime, with a random step size $h$, or
over the rationals. "No solution up to degree $d$" is therefore an exact negative result for that
degree (barring a vanishingly unlikely unlucky prime), not a numerical impression.

---

## 1. New integrable maps

### 1.1 The lifted isotropic quartic oscillator is integrable

Take $H=\tfrac12\lvert p\rvert^2+\tfrac14\lvert q\rvert^4$ with $q,p\in\mathbb R^2$ (a particle in a
rotation-invariant quartic well), and lift with **one** auxiliary variable $y=\lvert q\rvert^2$:

$$\dot q=p,\qquad \dot p=-y\,q,\qquad \dot y=2\,q\cdot p\qquad(z=(q,p,y)\in\mathbb R^5).$$

Kahan's map of this field (one $5\times5$ linear solve per step) conserves, **exactly and for
every $h$**:

$$c_h=y-\lvert q\rvert^2+\tfrac{h^2}{4}\lvert p\rvert^2,\qquad M_h=\frac{q_1p_2-q_2p_1}{1+\tfrac14h^2y},\qquad E_h=\frac{\tfrac12\lvert p\rvert^2+\tfrac14y^2+\tfrac14h^2M_h^2}{1+\tfrac12h^2\lvert q\rvert^2+\tfrac1{16}h^6M_h^2},$$

and it preserves the measure

$$\frac{dz}{\big(1+\tfrac14h^2y\big)\Big(1+\tfrac12h^2\lvert q\rvert^2-\tfrac14h^4\big(\tfrac12\lvert p\rvert^2+\tfrac14y^2\big)\Big)}.$$

Three integrals plus a measure in five dimensions, plus the rotation symmetry: the map is
integrable. The evidence is independent and exact:

| Check | Result |
|---|---|
| identities with $h$ symbolic, at random rational points | all three integrals and the measure exact |
| numerical drift, 6 orbits $\times\,2\times10^4$ steps, $h=0.3,0.5,0.9$ | $\le6\times10^{-14}$ for all three integrals; measure defect $\le3\times10^{-15}$ |
| degree growth $d_k$, $k=1,\dots,7$ | $4,12,25,44,68,97,132$: second differences $5$–$6$, so **quadratic growth** |
| algebraic relations along orbits, degrees 2–6 | $3,18,60,150,316$: exactly the counts for an intersection of **three** quadrics, so orbits lie on 2D surfaces |

Where each integral comes from is itself instructive:

* $c_h$: the companion lemma (the factor $q$ moves affinely).
* $M_h$: the **block theorem with two speeds**. The block $(q,p)$ moves as
    $1\cdot(p,0)+y\cdot(0,-q)$. The two fields (shears) do **not** commute, but the second-derivative
    operator acts on $M$ as a scalar, which is all the proof needs (§2.1). The formula predicted
    by the theorem is exactly $M_h$.
* $E_h$: found by an exact Darboux-polynomial search. The measure's two factors
    $K\propto1+\tfrac12h^2\lvert q\rvert^2-\dots$ and $Y=1+\tfrac14h^2y$ suggested the cofactor
    $(K'/K)(Y'/Y)^2$. Exactly one non-trivial polynomial has that cofactor, and the ratio with its
    partner simplifies to $E_h$. For $M=0$ it reduces to the known 1-DOF formula
    $\tilde H/(1+\tfrac12h^2q^2)$.

**On the original variables.** Started on the shifted sheet
$y_0=\lvert q_0\rvert^2-\tfrac{h^2}4\lvert p_0\rvert^2$, the map is (by back-substitution) an
integrable, symmetric, linearly implicit map of $\mathbb R^4$ that conserves $M_h$ and $E_h$. The
original energy error stays bounded at $O(h^2)$. I have not found this map in the literature, but I
have not searched exhaustively. Integrable discretisations of central-force problems exist, but
not, as far as I know, one produced by Kahan's method on a quadratic lift.

**Generalisations.**

* In $\mathbb R^n$ with any $U(y)=\tfrac12u_2y^2+u_1y$, *every* angular-momentum component is
    conserved as $M_{ab}/(1+\tfrac12h^2U'(y))$ (drift $3\times10^{-13}$ in $\mathbb R^3$).
* **The family is integrable beyond the example.** Adding a harmonic term
    ($H=\tfrac12\lvert p\rvert^2+\tfrac12\lvert q\rvert^2+\tfrac14\lvert q\rvert^4$) and moving
    to $\mathbb R^3$ (a 7-dimensional map) both give exactly the same quadratic degree sequence
    $4,12,25,44,68,97,132$. The control case, the anisotropic potential $(q_1^2+2q_2^2)^2$ that
    breaks the rotation symmetry, grows exponentially ($4,16,58,208,742,>900$). So the symmetry
    is what makes the map integrable.

### 1.2 Most lifted maps are not integrable, and the test tells them apart

| Kahan map | Degrees $d_1,\dots,d_7$ | Verdict |
|---|---|---|
| quartic oscillator lift, 3D (calibration) | $3,9,19,33,51,73,99$ | quadratic: integrable (known) |
| isotropic quartic lift, 5D (also with a harmonic term, and in $\mathbb R^3$, 7D) | $4,12,25,44,68,97,132$ | quadratic: **integrable (new)** |
| anisotropic quartic $(q_1^2+2q_2^2)^2$ lift, 5D | $4,16,58,208,742,>900$ | exponential (ratio $\approx3.6$): not integrable |
| non-symmetric quartic lift, 3D | $3,9,23,57,139,337,815$ | $d_{k+1}=2d_k+d_{k-1}+2$, growth $(1+\sqrt2)^k$: not integrable |
| open Toda, 3 particles, $(p,u)$ variables, 5D | $3,9,23,57,139,337,815$ | not integrable, despite a measure and two integrals |

So Kahan–Toda is a measure-preserving map with two integrals that is still chaotic. The
non-symmetric quartic preserves an area form on each leaf (previous note) but is chaotic within
the leaf, which is why its energy error is bounded but not constant.

### 1.3 How to look for more

The isotropic example suggests a recipe:

1. Choose a system with a continuous symmetry and a lift that respects it.
2. Use the block theorem to get the modified momentum.
3. Run a Darboux-polynomial search for the rest. The densities are a good source of cofactors.
4. Confirm integrability with the degree test.

Natural next candidates:

* the $\mathbb R^3$ isotropic map, now known to be integrable: count its integrals (it may be
    superintegrable, as the continuous system is) and find $E_h$ for general $U$;
* lifts of the integrable Hénon–Heiles and quartic cases (with suitable auxiliary variables);
* lifted tops with quartic Hamiltonians.

---

## 2. New preserved measures and integrals

| Map | New invariant | Status |
|---|---|---|
| isotropic quartic lift in $\mathbb R^n$ | $M_{ab}/(1+\tfrac12h^2U'(y))$ for all $a<b$ | **proved** (block theorem, two speeds) |
| isotropic quartic lift in $\mathbb R^2$ | $E_h$ and the measure of §1.1 | exact with $h$ symbolic at random rational points |
| open Toda, any $n$, $u_k=e^{q_k-q_{k+1}}$ | the **exact** Liouville measure $dp\,du/(u_1\cdots u_{n-1})$, independent of $h$; modified energy $E-\tfrac{h^2}8\lvert\dot p\rvert^2$ | **proved** (three shears, below); unique density for $n=3,4,5$ |
| every 3D quadratic Nambu system $\nabla H\times\nabla C$ | a measure $dz/P$ with $P$ of degree 4 | **conjecture**: exact for random $H$, $C$, for four quartic 1-DOF lifts, and closed form for all $h$ in one case |
| resonant blocks | modified invariants with squared denominators | proved (previous note) |

### 2.1 A correction that makes the block theorem stronger

The several-speeds version needs only that the second-derivative operators $\mathcal T_{kl}$
**map the invariants to invariants and commute as matrices**. The vector fields $L_k$ themselves
need not commute: the proof only ever applies the invariants to $w\mp\sum_ka_kL_k(w)$.

In the isotropic example $L_1=(p,0)$ and $L_2=(0,-q)$ generate $\mathfrak{sl}(2)$, yet
$\mathcal T_{12}M=M$ and $\mathcal T_{11}M=\mathcal T_{22}M=0$. The theorem then gives
$M/(1+a_1a_2)$ with $a_1a_2=\tfrac{h^2}4\cdot1\cdot y$.

The previous note wrongly attributed the failure for "opposite scaling" constraints to
non-commuting fields. The real reason is that $\mathcal T_{11}c=2x_ix_j$ is not an invariant, so
closure fails. That note is now corrected.

### 2.2 Why Kahan–Toda preserves the exact Liouville measure (proof sketch)

Write $x=\log u$ and $\Delta=Dp$ (so $\Delta_k=p_k-p_{k+1}$). Kahan's equations
$p'-p=\tfrac h2L(u+u')$ and $u_k'(1-\tfrac h2\Delta_k)=u_k(1+\tfrac h2\Delta_k')$ factor as three
shears:

1. $\hat x=x-\log(1-\tfrac h2\Delta)$, a shift of $x$ by a function of $p$;
2. $p'=p+h\big(\mathrm{Id}-\tfrac{h^2}4L\,\mathrm{diag}(e^{\hat x})\,D\big)^{-1}L\,e^{\hat x}$, a shift of $p$ by a function of $\hat x$;
3. $x'=\hat x+\log(1+\tfrac h2\Delta')$, a shift of $x$ by a function of $p'$.

Each shear preserves $dp\,dx=dp\,du/\prod u_k$. The same argument works for any potential
$\sum_kc_ke^{\alpha_k\cdot q}$ written in the variables $u_k=e^{\alpha_k\cdot q}$. The map is *not*
Poisson (defect $0.2$–$0.8$), so this is volume preservation, not symplecticity.

### 2.3 The Nambu measure conjecture

For every 3D Nambu system $\dot z=\nabla H\times\nabla C$ with quadratic $H$ and $C$, Kahan's map
appears to preserve $dz/P$ with $P$ a polynomial of degree 4. Known special cases:

* $H$ and $C$ homogeneous: CMMOQ 2014;
* $H$ and $C$ two-variable functions sharing one variable: CMOQ 2019, Theorem 2.

My exact searches find a unique degree-4 density for random inhomogeneous $H$ and $C$. Every 1-DOF
lift with **one** auxiliary variable is such a Nambu system: all potentials $\tfrac12p^2+V(q)$
with $V$ quartic (lift $y=q^2$) and the family $U(\varphi)+\kappa$. So a proof would show that
**the back-substituted Kahan method is area-preserving for all of these**: a linearly implicit,
symplectic-like integrator. (Not every quartic 1-DOF Hamiltonian has such a lift; your paper's
$\tfrac12p^2+q^2p^2+q^4$ does not.) You may know whether your 2019 discrete-Darboux
paper already covers inhomogeneous Nambu systems; I did not find it there.

---

## 3. Semi-discretised PDEs

| PDE class (after spatial discretisation) | Lift | What lifted Kahan conserves | Verdict |
|---|---|---|---|
| cubic wave equations: $\varphi^4$ / Klein–Gordon, FPU-$\beta$, nonlinear strings | $y_k=u_k^2$ or $r_k=(u_{k+1}-u_k)^2$ | every constraint, exactly (lemma); back-substitution gives a one-step method on $(u,v)$ | **useful, but no modified energy and no polynomial measure** (2-site $\varphi^4$: none up to degree 6; FPU with 2 masses: none up to degree 6) |
| Toda lattices (exponential interactions) | $u_k=e^{q_k-q_{k+1}}$ | modified energy, exact Liouville measure, no constraints needed | **fully protected** (not integrable for $n\ge3$) |
| rotor / XY chains, $-J\sum\cos(q_{k+1}-q_k)$, open ends | $(\sin\theta_k,\cos\theta_k)$ of the bond angles | modified energy (drift $10^{-15}$) and every modified circle | **fully protected**, but stable only for $h\omega\lesssim2$ when the stiffness enters through the nonlinear terms (below) |
| on-site trigonometric potential plus harmonic coupling (sine-Gordon) | $(q,\sin q,\cos q)$ | circles and energy, but $s=\sin q$ is not algebraic and drifts in phase | first note: phase drift |
| $O(n)$-symmetric coupled lattices | $y_k=\lvert u_k\rvert^2$ | constraints; the internal charge is **not** protected (drift $0.1$) | angular momentum does not survive coupling |
| nonlinear Schrödinger | $\rho_k=\lvert\psi_k\rvert^2$ | constraints are "opposite scaling", with no partner for 2 or more sites; no density up to degree 4 | not protected |
| parabolic (Allen–Cahn, Cahn–Hilliard) | $y=u^2$ | constraints are conditional (the reaction term is cubic) | not protected; use the attracting-sheet design of the first note |
| degree $\ge6$ nonlinearities | hierarchies | base constraint only | slow drift (previous note) |

**What you get for cubic wave equations** (the main PDE class):

* a linearly implicit, symmetric, second-order one-step method;
* one sparse solve per step, with local coupling in the $y$-equations;
* the linear part treated by a Cayley transform, so no time-step limit *from the linear part*;
* constraints that never drift.

**Protection is not stability.** A modified invariant such as
$(s^2+c^2)/(1+\tfrac14h^2\dot\theta^2)$ or $\tilde H/(1+\tfrac12h^2q^2)$ does not bound the
solution when $hA$ is large: the lifted variables may grow together with the speed. On a stiff
rotor chain ($N=64$, $J=1/\Delta x^2$, linear frequencies up to $\omega=20$; `rotor_chain_long.py`):

| $h\omega$ | lifted Kahan, $t\le600$ | Störmer–Verlet, $t\le20$ |
|---|---|---|
| 1 | energy error $5\times10^{-5}$, $E_h$ drift $2\times10^{-15}$ | $3\times10^{-4}$ |
| 2 | energy error $2\times10^{-4}$, $E_h$ drift $1\times10^{-15}$ | $2\times10^{-2}$ (its stability limit) |
| 4 | blows up at $t\approx118$ | blows up |
| 6 | blows up at $t\approx37$ | blows up |

Here the stiffness enters through the bilinear terms $c_k\dot\theta_k$, which Kahan treats half
explicitly. In $\varphi^4$/Klein–Gordon lattices the stiff Laplacian sits in the *linear* part,
which is why the first note found them stable well beyond the explicit limit. **Rule: put the
stiff terms in the linear part of the lift.**

**What you do not get:** a conserved energy or a preserved measure. Energy errors were bounded in
the first note's experiments, but there is no theorem behind that, and the negative measure
searches suggest none is coming.

**The honest competitor comparison**, which a PDE paper must contain:

| Method | Steps | Solve per step | Conserved for cubic wave equations |
|---|---|---|---|
| midpoint MQAV (your paper) | one-step | nonlinear | energy and constraints exactly |
| polarised schemes (Celledoni, McLachlan, McLaren, Owren, Quispel 2015) | **two-step** | linear | a modified energy and a measure (constant Poisson structure) |
| SAV / IEQ | one- or two-step | linear | a modified quadratic energy; auxiliary relation drifts |
| lifted Kahan (here) | **one-step** | linear | modified constraints; energy only for exponential and rotor lattices |
| Störmer–Verlet | one-step | explicit | symplectic, but limited by the stability condition $h\lesssim\Delta x$ |

So for $\varphi^4$-type PDEs, lifted Kahan's selling points are "one-step, linearly implicit,
consistent constraints, no CFL limit from the Laplacian", not energy. The structural win is for
Toda and rotor lattices, where it conserves a modified energy (and, for Toda, a measure) with one
linear solve, provided the time step resolves the nonlinear stiffness. Those are lattice models
rather than discretisations of the classical PDEs.

---

## 4. Your polarisation idea

The proposal: add $z_{ij}=x_i+x_j$ as a variable and set $y_{ij}=\tfrac12(z_{ij}^2-y_{ii}-y_{jj})$.

**Short answer: it cannot make every map Kahan-friendly.** The product constraint becomes a
combination of three square constraints:

$$y_{ij}-x_ix_j=\tfrac12\big[(w_{ij}-z_{ij}^2)-(y_{ii}-x_i^2)-(y_{jj}-x_j^2)\big],\qquad w_{ij}\approx z_{ij}^2,$$

so it is protected exactly when the three squares are. A square $y-x^2$ is protected only if it is
an *exact* integral of the lifted field. But $\tfrac{d}{dt}x^2=2x\dot x$ is **cubic** unless
$\dot x$ is affine in the lifted variables. A quadratic field can only reproduce it on the sheet,
which makes the constraint conditional. So you need $x_i$, $x_j$ and $x_i+x_j$ to move affinely,
which is the companion lemma's condition again.

**The coordinate-free reason.** Whether a quadratic integral is protected depends only on its
Hessian $S$ and on which linear combinations of the variables move affinely (the condition
$S\,Q=0$). The Hessian of $x_ix_j$ and of $\tfrac12\big((x_i+x_j)^2-x_i^2-x_j^2\big)$ is the
same matrix. Adding the linear variable $z_{ij}$ is just a linear change of coordinates, and
Kahan's method commutes with those. The automatic test of the previous note already searches over
all linear recombinations, since its rank test is coordinate-free.

**Demonstration** on your paper's cubic example $H=\tfrac12p^2+q^2p^2+q^4$, polarised with
$z=q+p$, $a=q^2$, $b=p^2$, $w=z^2$, $qp:=\tfrac12(w-a-b)$ (`polarization_nogo.py`):

| Constraint | Exact integral? | Kahan, $h=0.05$, first / last 10% of $2\times10^4$ steps |
|---|---|---|
| $z-q-p$ (linear) | yes | $2\times10^{-15}$ / $3\times10^{-12}$ |
| $a-q^2$, $b-p^2$, $w-z^2$ (squares) | **no** | $0.4$ / $1.6\times10^6$ |
| $\tfrac12(w-a-b)-qp$ (product) | **no** | $0.2$ / $7.5\times10^5$ |

**What linear recombination *is* good for:** finding two-variable members, as CMOQ do with
$X=x_1+x_4$. **What actually enlarges the protected class:**

* lift variables that move affinely (positions in mechanical systems);
* exponentials and single harmonics, whose forces are linear in the lifted variables;
* symmetric structures (block theorem);
* partner integrals.

Even the cleverest choice of lift variables only moves the problem one level up (the degree
regress of the previous note). For example, representing $q^2p$ by its own variable makes
$\dot q$ affine, but the new variable's constraint is then conditional.

---

## 5. What I would put in a paper

**Recommendation: one theory-first paper** built on the results that are proved or exact, with
the integrable map as its showcase. Working title: *Kahan's method on quadratic lifts: which
invariants survive, and new integrable maps.*

1. **Block theorem** (single and several speeds, with the corrected hypothesis). Corollaries:
    * CMOQ's two-variable theorem;
    * the companion lemma;
    * all normal-mode actions of generalised Suslov systems;
    * resonant blocks with squared denominators;
    * **modified angular momenta for isotropic systems in any dimension**.
2. **Protected span** and the classification of lift constraints: squares, affine factors,
    opposite scaling with partners. Include the no-go results (linear recombination such as
    polarisation cannot help; the degree regress) and the automatic test.
3. **Back-substitution:** lifted Kahan is a linearly implicit implementation of a symmetric
    one-step method on the original variables, with explicit formulas.
4. **The new integrable map** (§1.1), with its integrals, measure and degree growth, plus the
    controls in §1.2 to show the test discriminates.
5. **Measures:** the Toda Liouville measure (proof by shears) and the Nambu measure conjecture
    (or theorem, if the proof works). Its corollary: area preservation for every 1-DOF lift with
    one auxiliary variable (all quartic potentials).
6. **Numerics:**
    * long-time behaviour for the 1-DOF and isotropic examples;
    * a lattice section with $\varphi^4$/FPU (constraints protected), compared honestly with
        midpoint MQAV, polarised two-step schemes and SAV;
    * Toda and rotor chains (fully protected).

**Why this split.** Items 1–4 are proved or exactly verified and form a coherent story
("design the lift so the structure is protected, then back-substitute"). The isotropic map
gives the paper a result that integrable-systems readers will care about. Item 5 is the open
problem that would lift the paper from "useful" to "strong". If the Nambu proof resists, publish
it as a conjecture with the exact evidence, or make it a short second paper. **Venue:** J. Phys. A
or Nonlinearity for the integrable/measure angle (where the CMOQ series appeared); BIT or IMA
J. Numer. Anal. if the numerics and PDE comparison carry the weight.

**To do before writing:**

1. Try to prove the Nambu measure conjecture. The structure of the densities found
    (factorisation into CMOQ-type denominators in special cases; the $\tfrac{h^2}6V''$ term in
    the non-symmetric quartic) is the starting point.
2. Decide integrability for the isotropic variants (harmonic term, $\mathbb R^3$, anisotropic),
    and find the general formula for $E_h$ with $U(y)=\tfrac12u_2y^2+u_1y$.
3. Literature check on Kahan–Toda (measure and modified energy) and on integrable
    discretisations of the isotropic quartic oscillator.
4. Implement a polarised two-step scheme and SAV for the lattice comparison.
5. Write the computer-assisted proofs: exact identities in $h$, as in `isotropic_exact.py`.

---

## 6. Scripts

All in [`code-applications/`](code-applications/); its README maps each claim to a script.
