# How far does it go? Dimensions, affine factors, and the integrable class

*Fourth note in this folder (after [the first note](README.md), [the Kahan-invariants note](kahan-invariants.md)
and [the applications note](applications-and-paper-plan.md)). It answers three questions:*

1. *To what dimension do the arguments work?*
2. *Is there a trick to make the constraints' factors move affinely?*
3. *What is the most general class of ODEs that this method discretises integrably?*

*It also reports new numerics. Every claim is reproduced by a script in [`code-theory/`](code-theory/) (see §6).*

*Next note: [The Lie–Poisson draft, and whether (and why) to publish](lie-poisson-draft-and-publication.md).*

> **Main takeaway.** The mechanisms work in **every dimension**, but **integrability survives only
> when it comes from linear symmetries plus one degree of freedom**. In detail:
>
> * The isotropic quartic oscillator gives an integrable lifted Kahan map in every dimension and
>   for every quadratic $U$, with closed-form integrals and measure. The reason is that the map
>   keeps the motion in the plane of $q$ and $p$.
> * Every potential built from exponentials and single harmonics has a lifted Kahan map that
>   preserves the **exact Liouville measure** and a modified energy, in any dimension and with any
>   mass matrix.
> * In every lifted system tested, integrability that rests on *hidden* symmetries does not
>   survive: Toda with three or more particles, Garnier, the quartic $1{:}6{:}8$ and
>   $1{:}12{:}16$ cases, and even the separable $1{:}6{:}1$ case if the lift is not adapted to its
>   separating coordinates. This is not a general law: the genuinely quadratic Clebsch and
>   $\mathfrak{so}(4)$ Manakov tops keep hidden quadratic integrals under Kahan (see
>   [the next note](lie-poisson-draft-and-publication.md)).
>
> There are three tricks that make the constraints' factors move "affinely enough":
>
> * second-order (position) variables;
> * variables closed under differentiation (exponentials and harmonics);
> * fixed linear flows at affine speeds (the block theorem).
>
> Each buys exactly one level. Protecting the constraints is not enough for integrability: the
> original integrals must be protected too.

## Answers at a glance

| Question | Answer | Where |
|---|---|---|
| To what dimension? | **Every dimension** for the mechanisms (block theorem, lemma, protected span, the exponential/harmonic measure theorem). The isotropic integrable family works for every $n$: the map preserves the plane $\mathrm{span}\lbrace q,p\rbrace$, so $\mathbb R^n$ reduces to $\mathbb R^2$; closed forms are verified for $n=2,3,4$. **Invariant measures are special**: generic lifted lattices ($\varphi^4$, FPU, discrete NLS) have none. | §1 |
| A trick for affine factors? | Three tricks, each one level deep: positions in second-order form (and prolongation $v=\dot x$ for quadratic ODEs); lift functions closed under differentiation, which forces pure exponentials and harmonics; fixed linear flows with affine speeds. Linear changes of variables expose them. Polarisation and Sundman time changes do not help. | §2 |
| Most general integrable class? | Systems integrable through **linear symmetries plus one degree of freedom**, with a lift adapted to them: 1-DOF pencil and exponential/harmonic systems, decoupled sums, isotropic oscillators (any $n$; in the plane also with a magnetic field), translation-invariant two-body problems. Not, in every lifted case tested: anything needing hidden symmetries (genuinely quadratic exceptions such as the Clebsch top exist). | §3 |
| New numerics? | (i) A linearly implicit, integrable integrator for stiff isotropic anharmonic oscillators: stable at $h\omega=5$ where Störmer–Verlet blows up, with a quarter of the implicit midpoint rule's solves. (ii) Volume-preserving linearly implicit integrators for exponential and rotor lattices. | §4 |

---

## 0. Terms

As in the previous notes: $z$ is the lifted state, $\dot z=F(z)=Q(z,z)+Bz+b$, a subscript $h$ marks
the modified invariant that Kahan conserves exactly, and **degree growth** (algebraic entropy) is
the integrability test: polynomial growth of the degrees of iterates means integrable, exponential
growth means chaotic. "Factors" of a quadratic constraint are the linear forms in the range of its
Hessian $S$; the companion lemma needs $S\,Q=0$, i.e. *the factors move affinely*.

---

## 1. To what dimension?

### 1.1 The mechanisms are dimension-free

The proofs of the companion lemma, the block theorem (one or several speeds) and the protected
span never use the dimension. The same holds for the new measure theorem of §2.3. What does
depend on dimension is whether *enough* invariants are protected, which is the integrability
question of §3.

### 1.2 The isotropic family works in every dimension

Take $H=\tfrac12\lvert p\rvert^2+U(\lvert q\rvert^2)$ with $U(y)=\tfrac12u_2y^2+u_1y$,
$q,p\in\mathbb R^n$, lift $y=\lvert q\rvert^2$:

$$\dot q=p,\qquad \dot p=-2U'(y)\,q,\qquad \dot y=2\,q\cdot p .$$

**Why every dimension reduces to two.** Kahan's map is equivariant under the rotations
$(q,p,y)\mapsto(Rq,Rp,y)$. A rotation that fixes the plane $\mathrm{span}\lbrace q,p\rbrace$
pointwise fixes $z$, hence fixes $z'$. So $q'$ and $p'$ lie in the same plane, exactly as for the
continuous central-force motion. (Checked: in $\mathbb R^4$ the matrix $[q,p,q',p']$ has rank 2.)
Consequences:

* every result for $n=2$ holds for all $n$;
* one step costs the same as in $\mathbb R^2$ if it is computed in the plane.

**Closed forms for every $n$ and every quadratic $U$** (exact to $10^{-14}$ for $n=2,3,4$, several
$u_2,u_1$; `isotropic_general.py`, `isotropic_measure_nd.py`):

$$c_h=y-\lvert q\rvert^2+\tfrac{h^2}4\lvert p\rvert^2,\qquad M_{ab,h}=\frac{q_ap_b-q_bp_a}{1+\tfrac12h^2U'(y)},$$

$$E_h=\frac{\tfrac12\lvert p\rvert^2+U(y)+\frac{u_1^2}{2u_2}+\tfrac12h^2u_2\lvert M_h\rvert^2}{1+h^2u_2\lvert q\rvert^2+\tfrac14h^6u_2^2\lvert M_h\rvert^2},\qquad \lvert M_h\rvert^2=\sum_{a<b}M_{ab,h}^2,$$

and the measure

$$\frac{dz}{\big(1+\tfrac12h^2U'(y)\big)^{n-1}\Big(1+h^2u_2\lvert q\rvert^2-\tfrac12h^4u_2\big(\tfrac12\lvert p\rvert^2+U(y)+\tfrac{u_1^2}{2u_2}\big)\Big)}.$$

The linear term $u_1$ enters only through the shift $y\mapsto y+u_1/u_2$. That shift is an affine
change of variables, which Kahan respects. So the case $u_1\ne0$ is the case $u_1=0$ on a different
level set of $c_h$. Degree growth is quadratic for $n=2,3,4$ (§3).

### 1.3 Measures: a few structured classes in any dimension, otherwise none

| Lifted system | Polynomial invariant density? |
|---|---|
| any 3D quadratic Nambu system (all 1-DOF lifts with one auxiliary variable) | yes, degree 4 (conjecture, exact evidence) |
| isotropic quartic, any $n$ | yes, explicit (above) |
| exponential and harmonic potentials, any $n$, any mass matrix | **yes, the exact Liouville measure** (§2.3, proved) |
| $\varphi^4$ lattice (2 sites), FPU (2 masses), discrete NLS (2 sites) | **no** (exact searches up to degree 6, 6 and 4) |

So beyond three dimensions, measure preservation is a structural property (symmetry, or
exponential/harmonic variables), not a generic one.

---

## 2. A trick to make the factors move affinely

### 2.1 What protection actually needs

A constraint (or energy) with Hessian $S$ is protected when one of these holds:

* (lemma) $S\,Q=0$: its factors move affinely;
* (two-variable) it depends on two linear forms that move as (affine speed) × (rotation);
* (block theorem) its variables move along fixed linear flows at affine speeds, with the
    second-derivative operators closing up and commuting.

Every trick below manufactures one of these.

### 2.2 Trick 1: use positions, i.e. a second-order form

In a mechanical system the positions move affinely ($\dot q=M^{-1}p$). So every lift by products
of positions is protected, in any dimension. That covers all quartic potentials; forces can even
be quartic in $q$ (e.g. $\dot p=-y^2$), but then the energy is no longer quadratic in $z$.

**Prolongation** applies the same idea to a first-order quadratic ODE $\dot x=f(x)$: add
$v=\dot x$. Then $\dot x=v$ is affine and $\dot v=f'(x)v$ is quadratic. The constraint $v-f(x)$ is
exact and lemma-protected, so Kahan conserves $v-f(x)+\tfrac{h^2}8D^2f(x)[v,v]$ exactly
($3\times10^{-15}$ on the Euler top). **But** the original integrals become merely conditional for
the prolonged system. The prolonged Euler-top map is *not* integrable (degrees
$4,16,52,160,>400$), although plain Kahan on the Euler top is ($3,9,19,33,51,73$). Protecting the
constraints is not enough; the original integrals must be protected too. For cubic $f$,
prolongation hits the degree regress again: one level only.

### 2.3 Trick 2: lift functions closed under differentiation (a new measure theorem)

If the lifted variables $g_k(q)$ satisfy $\partial g_k/\partial q_j\in\mathrm{span}\lbrace g\rbrace$,
then $\dot g_k$ is bilinear in $(g,p)$, the force $-\nabla V$ is *linear* in $g$, so the momenta
move affinely and the energy is lemma-protected. Finite-dimensional function spaces closed under
differentiation consist of exponential polynomials. Polynomial factors such as $qe^q$ reintroduce
a relation ($g=u\log u$) that is only conditional and drifts. That leaves **pure exponentials and
single harmonics**:

$$H=\tfrac12p^\top Mp+\sum_kc_ke^{\alpha_k\cdot q}+\sum_jg_j\cos(\beta_j\cdot q),\qquad u_k=e^{\alpha_k\cdot q},\quad (c_j,s_j)=(\cos,\sin)(\beta_j\cdot q).$$

> **Theorem.** For any mass matrix $M$, any number of exponentials and harmonics (independent or
> not) and any dimension, Kahan's method in these variables conserves
> $E_h=E-\tfrac{h^2}8\dot p^\top M\dot p$ and every modified circle
> $(c_j^2+s_j^2)/(1+\tfrac14h^2\dot\theta_j^2)$, and it **preserves the canonical Liouville measure**
>
> $$dp\ \prod_k\frac{du_k}{u_k}\ \prod_j\frac{dc_j\,ds_j}{c_j^2+s_j^2}.$$

*Proof sketch.* In the coordinates $(p,\ \log u_k,\ \log\lvert\zeta_j\rvert,\ \arg\zeta_j)$, with
$\zeta_j=c_j+is_j$, one Kahan step factors into three shears:

1. each $\log u_k$ (and each $\zeta_j$) shifts by a function of $p$;
2. $p$ shifts by a function of the intermediate variables, because Kahan's momentum equation is
    affine in $p'-p$ with coefficients depending only on them;
3. each $\log u_k$ and $\zeta_j$ shifts again by a function of $p'$.

Each shear preserves volume in these coordinates. $\square$

*Checks* (`exp_trig_measure.py`, `exp_trig_measure_exact.py`): defect $\le4\times10^{-15}$ for open
Toda ($n=4$), periodic Toda (dependent exponents), coupled pendula with a non-diagonal mass matrix,
and a mixed exponential + cosine system in 3 DOF with a random mass matrix. The last is also exact
in rational arithmetic.

### 2.4 Trick 3: fixed linear flows at affine speeds, and linear changes of variables

The factors need not move affinely. It is enough that they move along a **fixed linear flow at an
affine speed**, possibly several. This is how isotropic systems get protected angular momenta:
$\dot q=1\cdot p$, $\dot p=2U'(y)\cdot(-q)$. Linear changes of variables are free, because Kahan
commutes with them, and they can expose such a structure:

* a mass matrix $M$ with potential $U(q^\top M^{-1}q)$ is isotropic after $q\mapsto M^{-1/2}q$;
* a separable system needs the lift in its separating coordinates (§3: $1{:}6{:}1$).

### 2.5 What no trick does: the one-level rule

Each trick protects one layer of variables: the positions, the exponentials, or the block. Any
constraint whose factor is itself a lifted variable with a quadratic velocity is conditional:
$\tfrac{d}{dt}(\text{factor})^2$ is cubic. Hence:

* sextic and higher polynomial potentials, and generic polynomial ODEs, cannot be fully protected;
* polarisation $z=x_i+x_j$ only rewrites the same Hessian (previous note);
* a Sundman time change $dt=g\,d\tau$ multiplies the field by $g$ and destroys its quadratic
    form.

The sextic isotropic oscillator illustrates this. Its natural closed lift (with
$y=\lvert q\rvert^2$, $w=y^2$, $\eta=q\cdot p$, $\zeta=\lvert p\rvert^2$) has a protected
angular momentum and a protected $w-y^2$, but its other constraints are conditional, and the map
is chaotic (degrees $5,16,43,114,305,>400$).

---

## 3. The most general integrable class

### 3.1 Experiments (exact degree growth, `degree_cases.py`)

| Lifted Kahan map | $d_1,\dots,d_6$ | Integrable? | Source of continuous integrability |
|---|---|---|---|
| isotropic quartic, $n=2,3,4$ (5D, 7D, 9D) | $4,12,25,44,68,97$ | **yes** | rotations + 1 DOF |
| isotropic quartic + harmonic term | $4,12,25,44,68,97$ | **yes** | rotations + 1 DOF |
| isotropic quartic in a uniform magnetic field | $4,13,28,49,76,109$ | **yes** | rotations + 1 DOF |
| decoupled $q_1^4+q_2^4$ | $5,17,37,65,101,145$ | **yes** | decoupling |
| $1{:}6{:}1$, lift adapted to $q_1\pm q_2$ | $5,17,37,65,101,145$ | **yes** | decoupling after a rotation |
| $1{:}6{:}1$, diagonal lift $y_k=q_k^2$ | $5,25,113,>400$ | no | same system, wrong lift |
| two-particle Toda; two coupled rotors | $2,4,\dots,12$; $3,9,19,33,51,73$ | **yes** | translation + 1 DOF |
| quartic $1{:}6{:}8$ and $1{:}12{:}16$ | $5,25,113,>400$ | no | hidden quartic integral |
| Garnier ($\tfrac12\sum a_kq_k^2+\tfrac14\lvert q\rvert^4$, $a_1\ne a_2$) | $4,16,58,208,>400$ | no | separable in elliptic coordinates |
| open Toda, 3 particles | $3,9,23,57,139,337$ | no | Lax pair (hidden) |
| anisotropic $(q_1^2+2q_2^2)^2$; generic $1{:}3{:}2$ | exponential | no | not integrable |
| sextic isotropic | $5,16,43,114,305,>400$ | no | lift not fully protected |

### 3.2 The principle

> **Conjecture (supported by every case above).** Kahan's method on a quadratic lift is
> integrable when the original system is integrable through **linear symmetries and one degree of
> freedom**, provided that:
>
> 1. the lift is equivariant under the symmetries and adapted to any decoupling;
> 2. all constraints are protected (quartic or exponential/harmonic type);
> 3. the reduced 1-DOF problem lies in a Kahan-integrable class (the pencil family of the
>    Kahan-invariants note, or the exponential/harmonic class).
>
> In every lifted system we tested, integrability that rests on hidden (non-linear) symmetries is
> destroyed. This is not true of every quadratic system: the Kahan maps of the Clebsch and
> $\mathfrak{so}(4)$ Manakov tops are integrable thanks to hidden quadratic integrals (Petrera,
> Pfadler, Suris), so the "only if" direction must be restricted to lifts.

**Why this is plausible.** Every mechanism that produces exact modified integrals is
linear-algebraic: Kahan commutes with linear maps, and the modified integrals are built from
quadratic invariants of linear flows and their speeds. A linear symmetry therefore passes to the
map (rotations give $M_h$), and a 1-DOF problem has its energy protected. Hidden integrals, such as
Toda's $\operatorname{tr}L^3$, Garnier's Uhlenbeck integrals or the quartic integrals of $1{:}6{:}8$,
are of higher degree and are not invariants of linear flows, so nothing protects them. The
$1{:}6{:}1$ pair is the sharpest illustration: the same Hamiltonian is integrable or chaotic
depending only on whether the lift respects the linear change of variables that separates it.

**The magnetic case** shows the block theorem is sufficient but not necessary: its canonical
angular momentum fails the closure test, yet the map is integrable. An exact Darboux search finds
two invariant densities, of degrees 3 and 4. Their ratio is a rational integral, conserved to
$2\times10^{-16}$, that mixes $q\times v$ linearly with energy-like terms. None of the three
mechanisms predicts it, so explaining it, perhaps by a block theorem without closure, is an open
problem (`magnetic_search.py`, `magnetic_integral.py`).

### 3.3 What this means for the question "which ODEs?"

The class of ODEs with integrable lifted-Kahan discretisations is, up to linear changes of
variables, generated by:

* **1-DOF pieces:** $U(\varphi(x))+\kappa(x)$ (quadratic $U$, quadratic forms $\varphi,\kappa$;
    this contains all symmetric quartic oscillators); single-exponential or single-harmonic
    potentials (pendulum, two-particle Toda); and the Morse oscillator, whose energy is a
    two-variable integral;
* **gluing operations:** decoupled sums, and isotropic extension to $\mathbb R^n$ (in the plane
    also with a uniform magnetic field; higher dimensions untested).

Among lifted systems, everything that needs hidden symmetries fell outside in our tests, as do
potentials of degree above four.

---

## 4. New numerics

### 4.1 A linearly implicit integrable integrator for stiff isotropic oscillators

$H=\tfrac12\lvert p\rvert^2+\tfrac12\omega^2\lvert q\rvert^2+\tfrac14\lvert q\rvert^4$ in
$\mathbb R^3$, $\omega=10$, $t\le200$ (`isotropic_stiff_numerics.py`). The stiff harmonic term
sits in the *linear* part of the lift, which Kahan treats implicitly.

| $h\omega$ | lifted Kahan (one $7\times7$ linear solve per step) | implicit midpoint (Newton) | Störmer–Verlet |
|---|---|---|---|
| 0.5 | $E_h$ and $M_h$ exact ($10^{-15}$, $10^{-13}$); energy error $6\times10^{-4}$; 4,000 solves | energy error $2\times10^{-4}$; 16,000 solves | energy error $5\times10^{-2}$ |
| 5 | $E_h$ and $M_h$ exact; energy error $6\times10^{-2}$, bounded; 400 solves | energy error $3\times10^{-3}$; 1,600 solves | blows up |

So the lifted map is an integrable, symmetric, linearly implicit method that stays stable far
beyond the explicit limit, at about a quarter of the midpoint rule's linear solves. At large
$h\omega$ its accuracy is lower than the midpoint rule's but bounded. Because the motion is
planar, the step can be done in the plane of $q,p$ at fixed cost in any dimension.

### 4.2 Volume-preserving linearly implicit integrators for exponential and rotor lattices

The theorem of §2.3 makes lifted Kahan an exactly volume-preserving (Liouville) one-step method
with a conserved modified energy for Toda chains, rotor/XY chains, coupled pendula and mixtures of
these, with one linear solve per step. The applications note showed that on a stiff rotor chain
this is long-time stable for $h\omega\le2$.

---

## 5. What this changes in the paper plan

The paper of the applications note gains a clean organising principle:

* **one theorem per trick**: lemma (positions), the Liouville-measure theorem (exponentials and
    harmonics), the block theorem (linear flows);
* **one conjecture for integrability** (linear symmetries plus one DOF), backed by the table in
    §3.1;
* **the isotropic family in closed form for all $n$** as the showcase.

The open problems become:

1. Prove the 3D Nambu measure conjecture (it covers all 1-DOF lifts with one auxiliary variable).
2. Explain the magnetic case, i.e. extend the block theorem beyond closure.
3. Prove the integrability conjecture, at least the "linear symmetry plus one DOF implies
    integrable" direction, perhaps via reduction of the equivariant map.

---

## 6. Scripts

All in [`code-theory/`](code-theory/); its README maps each claim to a script.
