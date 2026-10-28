# Which lift constraints can Kahan's method protect?

*Follow-up to [Quadratic lifts beyond the midpoint rule](README.md). Your question: "find even
more invariants of Kahan maps that we can use to make Kahan-friendly discretisations, like the
companion lemma about affine-drift constraints. Ideally we would have a way to express all
constraints $y-x_ix_j$ as a modified Kahan integral so that we can back-substitute this into the
lifted Kahan method." Every numerical claim is reproduced by a script in
[`code-invariants/`](code-invariants/) (see [§10](#10-reproducibility)).*

*Next note: [What the Kahan-lift invariants are good for, and what to write up](applications-and-paper-plan.md)
(a new integrable map, new measures, PDEs, why polarisation does not help, and a paper plan).*

> **Main takeaway.** Kahan's method keeps an exactly conserved, slightly modified copy of every
> quadratic invariant in a *linear space* of "protected" invariants. That space is spanned by
> three kinds of invariant:
>
> * invariants of two variables (Celledoni–McLaren–Owren–Quispel 2019);
> * invariants whose curvature only involves variables that move affinely (the companion lemma);
> * invariants of blocks of variables that move along fixed linear flows at affine speeds (a new
>   **block theorem**, which contains the other two).
>
> So a lift constraint $y-x_ix_j$ becomes a modified Kahan integral, namely
> $y-x_ix_j+\tfrac{h^2}{4}\dot x_i\dot x_j$, whenever its two factors move affinely, and this
> always holds for squares $y-x_i^2$. When the factors scale in opposite directions, the
> constraint needs a *partner* integral, such as the energy. Constraints that hold only on the
> constraint sheet are never protected. Whenever the modified constraint can be solved for $y$,
> the lifted Kahan method is a linearly implicit way to run a symmetric one-step method on the
> original variables. For quartic one-degree-of-freedom potentials that method also preserves an
> area form: proved in the integrable cases, verified exactly in the others.

## Results at a glance

| Question | Answer | Where |
|---|---|---|
| Can *every* constraint $y-x_ix_j$ be written as a modified Kahan integral? | **Exactly-conserved constraints:** yes when both factors move affinely (always for squares). If the factors scale in opposite directions, only with a partner integral. **Conditional constraints** (true only on the sheet): never. For generic polynomial ODEs some constraint ends up conditional (a rule of thumb), but triangular chains of constraints drift only slowly. | §2 |
| More invariants of Kahan maps? | A block theorem: a block of variables moving along a fixed linear flow at one affine speed keeps modified versions of **all** quadratic invariants of that flow. In resonant (non-diagonalisable) blocks this gives modified invariants with squared denominators, which the two-variable theorem cannot produce. | §1.3 |
| Can a constraint borrow protection? | Yes. Protection survives linear combinations, so $c=(I_1-I_2)/(\lambda_1-\lambda_2)$ is protected whenever the "pencil members" $I_k=\tilde H+\lambda_kc$ are. An automatic test decides this for any lift. | §1.4, §7 |
| A family where every constraint has a partner? | Lifts of 1-DOF Hamiltonians $U(\varphi(x))+\kappa(x)$. They are integrable, and they turn out to be Celledoni–McLaren–Owren–Quispel's integrable Nambu family after a linear change of variables. | §3 |
| Back-substitution? | The lifted Kahan method started on the "shifted sheet" *is* a symmetric one-step method on the original variables. For mechanical lifts $y=qq^\top-\tfrac{h^2}{4}vv^\top$ explicitly. | §4 |
| New invariant measures? | For every quartic 1-DOF potential tested (lift $y=q^2$), Kahan preserves $dz/P$ with $P$ a degree-4 polynomial. The example has a closed form valid for all $h$; the general case is now proved, with a closed form for every quartic potential (paper, Proposition 5.3). Consequence: the back-substituted method preserves an area form, which explains the bounded energy error. | §6 |
| Kahan-friendly classes beyond polynomials? | Toda-type exponential potentials (no constraints needed), Morse, trigonometric potentials with a constant mass matrix, and generalised Suslov systems (all normal-mode actions). | §5 |

---

## 0. Abbreviations and notation (read this first)

**Abbreviations.**
ODE / PDE = ordinary / partial differential equation.
DOF = degree of freedom.
MQAV = multiple quadratic auxiliary variables (your lifting method).
CMOQ = Celledoni, McLaren, Owren, Quispel (2019), *Geometric and integrability properties of Kahan's method: the preservation of certain quadratic integrals*.
CMMOQ = Celledoni, McLachlan, McLaren, Owren, Quispel (2014), *Integrability properties of Kahan's method*.
KAM = Kolmogorov–Arnold–Moser (theory of invariant curves for near-integrable maps).
FPU = Fermi–Pasta–Ulam. KG = Klein–Gordon. BH = Bose–Hubbard.

**Notation.**

| Symbol | Meaning |
|---|---|
| $x\in\mathbb R^n$, $y$ | original variables; auxiliary ("lifted") variables, e.g. $y=x_ix_j$ |
| $z=(x,y)\in\mathbb R^N$ | state of the lifted system |
| $\dot z=F(z)=Q(z,z)+Bz+b$ | lifted quadratic vector field: $Q$ symmetric bilinear (the quadratic part), $B$ a matrix, $b$ a vector |
| $F'(z)$ | Jacobian matrix of $F$ |
| $h$, $z\mapsto z'$ | time step; one Kahan step |
| $I(z)=\tfrac12z^\top Sz+s^\top z$ | a quadratic function: $S$ is its Hessian matrix (its "curvature"), $s$ its linear part |
| integral (first integral, invariant) | a function with $\nabla I\cdot F=0$, so it is constant along exact solutions |
| $I_h$ | **modified integral**: $I_h=I+O(h^2)$ and $I_h(z')=I_h(z)$ *exactly* for Kahan's map. A subscript $h$ always means this. |
| protected | an integral that has a modified integral |
| $\tilde H$ | the lifted energy (a quadratic in $z$ that equals $H$ on the sheet) |
| $c=y-x_ix_j$ | a lift constraint; the constraint sheet is $\lbrace c=0\rbrace$ |
| exact constraint | $c$ is an integral everywhere in $z$-space |
| conditional constraint | $\dot c=0$ only on the sheet $c=0$ |
| affine function | polynomial of degree at most one, e.g. $A(z)=2+z_1-3z_4$ |
| $J$ | the quarter-turn on $\mathbb R^2$: $J(a,b)=(b,-a)$ |

**Kahan's method** replaces every quadratic term by "half old × new + half new × old":

$$\frac{z'-z}{h}=Q(z,z')+B\,\frac{z+z'}{2}+b\qquad\Longleftrightarrow\qquad z'=z+h\Big(\mathrm{Id}-\tfrac h2F'(z)\Big)^{-1}F(z).$$

The new state appears linearly, so a step is **one linear solve** ("linearly implicit").

**Running example.** The quartic oscillator $H=\tfrac12p^2+\tfrac14q^4$ with $y=q^2$ has the
lifted field $\dot q=p$, $\dot p=-qy$, $\dot y=2qp$. The first note showed that Kahan conserves
$c_h=y-q^2+\tfrac{h^2}4p^2$ and $\tilde H_h=(\tfrac12p^2+\tfrac14y^2)/(1+\tfrac12h^2q^2)$.

---

## 1. Three ways Kahan's map protects a quadratic invariant

### 1.1 Mechanism A: integrals of two variables (CMOQ 2019, known)

If $I=\tfrac12a_1w_1^2+a_2w_1w_2+\tfrac12a_3w_2^2+a_4w_1+a_5w_2$ depends on two variables only,
the field moves them as $(\dot w_1,\dot w_2)=A(z)\,J\nabla I$ with an affine "speed" $A$, and
Kahan conserves

$$I_h=\frac{I+\tfrac18h^2D_2A^2}{1+\tfrac14h^2D_1A^2},\qquad D_1=a_1a_3-a_2^2,\qquad D_2=2a_2a_4a_5-a_3a_4^2-a_1a_5^2 .$$

Two things make this more widely applicable than it looks:

* **Coordinate-free test.** Kahan's method commutes with linear changes of variables (CMOQ use
    this too). So what matters is whether $I$ depends on two *linear combinations* of
    variables, which holds exactly when $\operatorname{rank}[\,S\mid s\,]\le2$ (the Hessian with
    the linear part appended as an extra column).
* **Complex members count.** If a complex combination of integrals passes the rank test, the
    formula holds over $\mathbb C$, and its real and imaginary parts are two real modified
    integrals.

*Example.* The quartic-oscillator energy $\tfrac12p^2+\tfrac14y^2$ uses only $(p,y)$, with $A=-2q$,
$D_1=\tfrac12$, $D_2=0$, giving $\tilde H_h$ above.

### 1.2 Mechanism B: the companion lemma (first note), seen as a telescoping product rule

**Lemma (first note, §3.2).** If $I$ is an integral and its Hessian annihilates the quadratic part
of the field ($S\,Q(u,v)=0$ for all $u,v$), then Kahan conserves
$I_h=I-\tfrac{h^2}{8}F^\top SF$. For a constraint $c=y-x_ix_j$ the condition says
"$\dot x_i$ and $\dot x_j$ are affine", and

$$c_h=y-x_ix_j+\frac{h^2}{4}\,\dot x_i\,\dot x_j .$$

**Why it works, in one line of algebra.** Take the running example. Kahan's equations give
$q'-q=\tfrac h2(p+p')$ and $y'-y=h(qp'+q'p)$, while the true change of $q^2$ is
$q'^2-q^2=(q'-q)(q'+q)=\tfrac h2(p+p')(q+q')$. The two disagree by

$$\tfrac h2(p+p')(q+q')-h(qp'+q'p)=\tfrac h2(p'-p)(q'-q)=\tfrac{h^2}{4}\big(p'^2-p^2\big),$$

which is itself a difference ("telescopes"), so $y-q^2+\tfrac{h^2}{4}p^2$ is conserved exactly.
The lemma says this always happens when the factors move affinely.

**It reproduces known results.** CMMOQ 2014 prove case by case that Kahan conserves
$z-6x^2+\tfrac32h^2y^2$ for the Ishii system ($\dot x=y$). The lemma gives this at once:
$S=-12\,e_xe_x^\top$, so $I_h=z-6x^2-\tfrac{h^2}8(-12\dot x^2)$.

**It also protects energies**, not only constraints. Whenever the energy's curvature sits in
variables with affine velocity, $\tilde H_h=\tilde H-\tfrac{h^2}8\dot\pi^\top G\dot\pi$ ($\pi$ =
momenta, $G$ = constant inverse mass matrix). This covers the Toda-type and trigonometric classes
of §5.

### 1.3 Mechanism C (new): the block theorem

**Setting.** Part of the state, a block $w\in\mathbb R^m$, moves as

$$\dot w=A(z)\,L(w),\qquad L(w)=Lw+\ell,$$

where $A$ is an affine scalar "speed" (it may depend on anything in $z$) and $L(w)$ is a *fixed*
affine vector field on the block (constant matrix $L$, constant vector $\ell$). Think of a bead
that always moves along the same family of curves, while the rest of the system controls how
fast it moves.

Let $V$ be the space of quadratic invariants of the fixed flow $\dot w=L(w)$ (include
constants). For $I\in V$ define

$$(\mathcal TI)(w)=L(w)^\top\,\nabla^2I\;L(w),$$

the second derivative of $I$ along the flow direction. $\mathcal TI$ is again an invariant of
the fixed flow, so $\mathcal T$ maps $V$ to itself. Pick a basis $\mathbf I=(I_1,\dots,I_r)$ of $V$
and let $T$ be the $r\times r$ matrix with $\mathcal TI_k=\sum_lT_{kl}I_l$.

> **Block theorem (single speed).** Kahan's method conserves every component of
>
> $$\mathbf I_h(z)=\Big(\mathrm{Id}+\frac{h^2A(z)^2}{8}\,T\Big)^{-1}\mathbf I(w).$$
>
> **Several speeds.** If $\dot w=\sum_kA_k(z)L_k(w)$ with affine speeds $A_k$ and fixed affine
> fields $L_k$, let $V$ be the common quadratic invariants and
> $\mathcal T_{kl}I=L_k^\top\nabla^2I\,L_l$. If each $\mathcal T_{kl}$ maps $V$ to itself and
> their matrices commute, Kahan conserves
> $\big(\mathrm{Id}+\tfrac{h^2}8\sum_{k,l}A_kA_lT_{kl}\big)^{-1}\mathbf I$. The vector fields
> $L_k$ themselves need **not** commute.

**Proof (short, because it is the heart of the matter).** Polarisation turns $A(z)L(w)$ into
$\tfrac12[A(z)L(w')+A(z')L(w)]$. With $a=\tfrac h2A(z)$ and $a'=\tfrac h2A(z')$, the block's
Kahan step is

$$w'-a\,L(w')=w+a'\,L(w).$$

Apply $I\in V$ to both sides. $I$ is quadratic and its derivative along $L$ vanishes, so only
the second-order term survives: $I(w\pm aL(w))=I(w)+\tfrac{a^2}{2}(\mathcal TI)(w)$. Hence

$$M(a)\,\mathbf I(w')=M(a')\,\mathbf I(w),\qquad M(a)=\mathrm{Id}+\tfrac{a^2}{2}T .$$

The speeds cross over: the speed at the old point multiplies the new point, and vice versa. All
the matrices $M(\cdot)$ are functions of the one matrix $T$, so they commute, and therefore
$M(a')^{-1}\mathbf I(w')=M(a)^{-1}\mathbf I(w)$. This is the claim, since
$\tfrac{a^2}2=\tfrac{h^2A^2}8$. With several speeds the argument is identical once the $T_{kl}$
commute. $\blacksquare$

**Intuition.** If $a=a'$, the step is exactly the midpoint rule (Cayley transform) for the fixed
linear flow, which conserves all its quadratic invariants. Kahan's only "error" is that the speed
is sampled at different ends. The error is an explicit speed-dependent rescaling, so dividing it
out at each end gives exact conservation.

**What it contains** (each line checked numerically to about $10^{-15}$):

| Block | Matrix $T$ | Modified invariants Kahan conserves |
|---|---|---|
| two variables, $\dot w=A\,J\nabla I$ (CMOQ) | $\mathcal TI=2D_1I-D_2$ | the CMOQ formula of §1.1 |
| variables with affine velocities: several speeds along commuting "translations" (the lemma) | nilpotent ($\mathcal T_{kl}I$ is a constant) | $I-\tfrac{h^2}8F^\top SF$ |
| rotation by a complex structure, $\dot w=A\,\mathbb Jw$ with $\mathbb J^2=-\mathrm{Id}$ | $2\,\mathrm{Id}$ on Hermitian forms (the quadratic forms $w^\top Sw$ with $S\mathbb J=\mathbb JS$) | every Hermitian form divided by $1+\tfrac14h^2A^2$ |
| generalised Suslov system $\dot x=\ell(x)\,\mathbb J_0\mathsf Hx$ in $\mathbb R^{2m}$ ($\mathbb J_0$ the standard symplectic matrix, $\mathsf H$ a constant symmetric matrix with frequencies $\omega_k$, $\ell$ linear) | $\operatorname{diag}(2\omega_k^2)$ on normal-mode actions | **each** action $I_k/(1+\tfrac14h^2\ell(x)^2\omega_k^2)$, i.e. $m$ integrals (CMMOQ 2014 had the total energy and a measure) |
| **resonant block** $\dot u=A(\omega\mathbb Ju+v)$, $\dot v=A\,\omega\mathbb Jv$ ($u,v\in\mathbb R^2$) | a Jordan block | $N_h=\dfrac{N}{1+\beta}$, $\ G_h=\dfrac{G}{1+\beta}+\dfrac{\beta N/\omega}{(1+\beta)^2}$, with $G=u^\top\mathbb Jv$, $N=\lvert v\rvert^2$, $\beta=\tfrac14h^2A^2\omega^2$ |
| nilpotent block $\dot w=A\,(w_2,w_3,0)$ | nilpotent | $w_2^2-2w_1w_3-\tfrac14h^2A^2w_3^2$ |
| isotropic lift $\dot q=1\cdot p$, $\dot p=2U'(y)\cdot(-q)$ in $\mathbb R^n$: two shear fields that generate $\mathfrak{sl}(2)$ (they do **not** commute) | scalar on angular momenta ($\mathcal T_{12}M=M$, $\mathcal T_{11}=\mathcal T_{22}=0$) | every $M_{ab}=q_ap_b-q_bp_a$ divided by $1+\tfrac12h^2U'(y)$ (see the [next note](applications-and-paper-plan.md)) |

**What is genuinely new here.** When the fixed flow is diagonalisable (over $\mathbb C$), every
quadratic invariant is a combination of two-variable invariants in normal-mode coordinates. The
block theorem then reproduces CMOQ plus a change of variables, packaged in one formula. Two
things are new:

1. **Several speeds.** This is where the lemma comes from: products $y-x_ix_j$ with affine factors
    are invariants of two flows whose second-derivative operators are nilpotent (see §2.1). It also
    gives modified angular momenta for isotropic systems, where the flows do not even commute.
2. **Non-diagonalisable blocks.** Here invariants are protected that are neither two-variable
    combinations nor lemma-type. In the resonant example the automatic test of §7 reports
    $G=u^\top\mathbb Jv$ as "integral, NOT protected" by mechanisms A and B. Yet the block
    theorem's $G_h$, with its **squared denominator**, is conserved to $6\times10^{-16}$. The
    two-variable-style guess $G/(1+\beta)$ drifts by $10^{-4}$.

### 1.4 The protected span: constraints can borrow protection

If Kahan conserves $I_{1,h}$ and $I_{2,h}$, it conserves $\alpha I_{1,h}+\beta I_{2,h}$, which is a
modified version of $\alpha I_1+\beta I_2$. So the protected integrals form a linear space

$$\mathcal P=\operatorname{span}\big\lbrace\text{two-variable members (real and imaginary parts)},\ \text{lemma members},\ \text{block members}\big\rbrace .$$

A constraint that is not protected on its own is still protected if it lies in $\mathcal P$, for
example as the difference of two protected combinations of the energy and the constraint. That
is how the constraint in §3 gets protected. `protected_span.py` computes $\mathcal P$
automatically (§7).

---

## 2. Which lift constraints can be protected?

### 2.1 Exact constraints: a complete classification

Let $c=y-x_ix_j$ be an **exact** integral of the lifted quadratic field, and let $Q_i$, $Q_j$ be
the quadratic parts of $\dot x_i$, $\dot x_j$. Exactness means
$\dot y=\dot x_ix_j+x_i\dot x_j$ identically. The left side is quadratic, so the cubic terms on
the right must cancel: $Q_ix_j+x_iQ_j=0$. That leaves very few possibilities:

| Case | What exactness forces | Protected? | Modified constraint |
|---|---|---|---|
| square, $i=j$ | $2x_iQ_i=0$, so $\dot x_i$ is affine | **always** (two variables *and* the lemma) | $y-x_i^2+\tfrac{h^2}{4}\dot x_i^2$ |
| product, affine factors | $Q_i=Q_j=0$ | **always** (lemma; two commuting flows in the block theorem) | $y-x_ix_j+\tfrac{h^2}{4}\dot x_i\dot x_j$ |
| product, opposite scaling | $Q_i=x_iR$, $Q_j=-x_jR$ for a linear $R\ne0$ | only with a partner (it must lie in $\mathcal P$) | from the partner, e.g. §3 |

"Opposite scaling" means the two factors are stretched and squeezed at the same rate, like
$(q,p)\mapsto(e^{\tau}q,e^{-\tau}p)$, which keeps $qp$ fixed. In block-theorem language the block
$(x_i,x_j,y)$ moves along the scaling field $(x_i,-x_j,0)$ and the translation fields
$(1,0,x_j)$, $(0,1,x_i)$. The second derivative of $c$ along the scaling field is $2x_ix_j$, which
is **not** an invariant of the translations. Closure fails, which is why the theorem cannot
protect such a constraint alone. (An earlier version of this note blamed non-commuting fields;
that is not the reason, see §1.3.)

*Example.* In the "mixed" lift of $H=\tfrac12(qp)^2+\tfrac12(q^2+p^2)$ with $y=qp$, the field is
$\dot q=qy+p$, $\dot p=-py-q$, so $R=y$. The constraint is protected only because the energy is a
partner (§3).

### 2.2 Conditional constraints, and why generic ODEs cannot be fully protected

A conditional constraint (true only on the sheet) has no modified version: the paper's
Lie–Poisson lift of its non-separable example has no exact constraint integrals, and they drift.

**Rule of thumb (degree regress).** Lemma protection reaches one level deep: the *factors* must
have affine velocities. In mechanical systems the natural affine variables are the positions
($\dot q=M^{-1}p$), so the protected lifts use products and squares of positions. Those represent
potentials up to degree four, where the force $-\nabla V$ becomes quadratic in $z$. A degree-six
potential needs products of *lifted* variables ($q^6=y\cdot y^2$), but lifted variables have
quadratic velocities ($\dot y=2qp$). Their constraints are then either conditional or need
partners, and in general neither is available. The chain closes only when derivatives stay
inside a finite set of affine-moving functions: exponentials ($\tfrac{d}{dt}e^{q}=e^{q}\dot q$,
with force *linear* in $e^q$) and single harmonics (§5).

Rewriting products by polarisation, $x_ix_j=\tfrac12\big((x_i+x_j)^2-x_i^2-x_j^2\big)$ with
$z=x_i+x_j$ as an extra variable, does not escape this. The square constraints are exact only if
$x_i$, $x_j$ and $x_i+x_j$ all move affinely, which is the lemma's condition again. The protection
test is coordinate-free, so linear recombination cannot change its verdict (details and a
numerical demonstration in the [next note](applications-and-paper-plan.md#4-your-polarisation-idea)).

### 2.3 Triangular hierarchies: what happens beyond degree four

If the unprotected constraints are driven only by protected ones, the drift stays small. The
octic chain $H=\tfrac12p^2+\tfrac18q^8$ with $y_1=q^2$, $u=q^3$, $y_2=q^4$ satisfies

$$\tfrac{d}{dt}(u-qy_1)=2p\,(y_1-q^2),\qquad \tfrac{d}{dt}(y_2-y_1^2)=4p\,(u-qy_1).$$

The protected base constraint $y_1-q^2$ drives the next level, which drives the next. The energy
$\tfrac12p^2+\tfrac18y_2^2$ is a two-variable integral and is protected
(drift $1.1\times10^{-14}$ over $4\times10^5$ steps).

| Test (Kahan, no resetting) | Size of the unprotected constraint(s) | Energy error |
|---|---|---|
| octic chain, $h=0.05$, $4\times10^5$ steps | $\lvert u-qy_1\rvert\le5.3\times10^{-4}$, $\lvert y_2-y_1^2\rvert\le1.9\times10^{-3}$, **identical** in the first and last 10% of the run | $3.6\times10^{-4}$, bounded |
| octic chain, four more runs ($h=0.1$ and $0.2$, three initial conditions) | $2\times10^{-3}$ to $5\times10^{-2}$, again no growth | $1.4\times10^{-3}$ to $1.5\times10^{-2}$, bounded |
| $\varphi^6$ Klein–Gordon lattice ($N=128$, $w=u^2$, $s=u^3$), $h=\tfrac12\Delta x$, $T=400$ | $\lvert s-uw\rvert$: $5.2\times10^{-2}\to7.9\times10^{-2}$ (slow growth) | $2\%$, flat |
| same, $h=2\Delta x$ | $\lvert s-uw\rvert$: $0.55\to1.2$ | $16$–$17\%$, flat |
| same, $h=2\Delta x$, resetting $s:=uw$ every step | not applicable | $16\%\to33\%$: **resetting hurts** |

**Lesson.** In one DOF the hierarchy behaves as if protected. In lattices it drifts slowly, and
projecting it back by resetting destroys the structure that keeps the base protected. If you need
degree-six nonlinearities in a PDE, the midpoint MQAV method of your paper is still the robust
choice.

---

## 3. A family where the constraint needs a partner: one-DOF lifts

**Setting.** Write $x=(q,p)$ and consider

$$H(x)=U(\varphi(x))+\kappa(x),\qquad \varphi=\tfrac12x^\top\Phi x,\quad \kappa=\tfrac12x^\top Kx,\quad U(y)=\tfrac12u_2y^2+u_1y\ (u_2\ne0),$$

with symmetric $2\times2$ matrices $\Phi$, $K$. Lift with $y=\varphi(x)$ and
$\tilde H=U(y)+\kappa(x)$. This covers all symmetric quartic oscillators ($\varphi=q^2$) and the
mixed example ($\varphi=qp$). In $z=(q,p,y)$ the lifted system is the Nambu system
$\dot z=\nabla\tilde H\times\nabla c$ with $c=y-\varphi(x)$.

**The pencil trick.** Combine energy and constraint: $I_\lambda=\tilde H+\lambda c$. Its
$x$-part is $\tfrac12x^\top(K-\lambda\Phi)x$, so it becomes a two-variable integral exactly when
$K-\lambda\Phi$ has rank one, i.e. when

$$\det(K-\lambda\Phi)=0,$$

a quadratic equation for $\lambda$. If $\det\Phi=0$ there is also a root "at infinity", meaning
$c$ itself is two-variable. At a root, $K-\lambda\Phi=\mu\,\ell\ell^\top$. With $w=\ell\cdot x$:

$$I_\lambda=\tfrac\mu2w^2+\tfrac{u_2}2y^2+(u_1+\lambda)y,\qquad A_\lambda=\ell^\top J\Phi x,\qquad D_1=\mu u_2,\qquad D_2=-\mu(u_1+\lambda)^2,$$

$$I_{\lambda,h}=\frac{I_\lambda-\tfrac18h^2\mu(u_1+\lambda)^2A_\lambda^2}{1+\tfrac14h^2\mu u_2A_\lambda^2}.$$

**Result.** If the two roots $\lambda_1\ne\lambda_2$ are distinct (real, or a complex-conjugate
pair, in which case take real and imaginary parts), then

$$c_h=\frac{I_{\lambda_1,h}-I_{\lambda_2,h}}{\lambda_1-\lambda_2},\qquad \tilde H_h=\frac{\lambda_2I_{\lambda_1,h}-\lambda_1I_{\lambda_2,h}}{\lambda_2-\lambda_1}$$

are both conserved. Kahan also preserves the measure $dq\,dp\,dy/(D_{\lambda_1}D_{\lambda_2})$ with
$D_\lambda=1+\tfrac14h^2\mu u_2A_\lambda^2$, so the lifted Kahan map is **integrable**.

**Credit where due.** The Nambu field can be rewritten as
$\nabla\tilde H\times\nabla c=(\lambda_2-\lambda_1)^{-1}\nabla I_{\lambda_1}\times\nabla I_{\lambda_2}$,
and in the coordinates $(w_1,y,w_2)$ both $I_\lambda$ are two-variable functions sharing $y$. That
is precisely the integrable family of **CMOQ 2019, Theorem 2**, which also gives the product
measure. So this section is an application of their theorem. The new point is that MQAV lifts of
$U(\varphi)+\kappa$ land in that family after the change of variables given by the pencil. This
confirms the guess in the first note that the quartic oscillator "may sit inside their
10-parameter integrable family".

*Checks.* Exact rational one-step checks give zero deviation for real roots
$\lambda=(82\pm\sqrt{3001})/51$, a complex pair $\tfrac3{10}\pm i$, and the mixed example $\pm1$.
The measure defect is about $8\times10^{-16}$.

**Worked example (mixed).** $\varphi=qp$, $K=\mathrm{Id}$, $U=\tfrac12y^2$, so $\lambda=\pm1$. With
$u=(q+p)/\sqrt2$ and $v=(q-p)/\sqrt2$:

$$I_{+,h}=\frac{y+v^2+\tfrac12y^2-\tfrac14h^2u^2}{1+\tfrac12h^2u^2},\qquad I_{-,h}=\frac{u^2+\tfrac12y^2-y-\tfrac14h^2v^2}{1+\tfrac12h^2v^2},\qquad c_h=\tfrac12\big(I_{+,h}-I_{-,h}\big).$$

Orbits lie on two quadrics (relation counts 2, 8, 19, 36, 60 at degrees 2–6). Over $2\times10^5$
steps, $\max\lvert y-qp\rvert=1.1\times10^{-3}$ in both the first and the last 10% of the run.

**Outside the family.** A cubic term breaks the pencil. For $V=\tfrac14q^4+\tfrac13q^3+\tfrac12q^2$
the lifted energy contains $qy$, only the constraint is protected, and the energy is not. Section 6
shows that such maps still preserve a measure, which explains their bounded energy error.

---

## 4. Back-substitution: the lifted method *is* a method on the original variables

**Principle.** Suppose the modified constraint $c_h(x,y)=0$ can be solved for $y=\psi_h(x)$.
Kahan conserves $c_h$ exactly, so the "shifted sheet" $\lbrace c_h=0\rbrace$ is invariant. The
lifted Kahan map restricted to it is therefore a one-step map $x\mapsto x'$ on the original phase
space. It is second order and **symmetric**: Kahan's map is symmetric and the shifted sheet
depends only on $h^2$. You implement it by running the lifted method from $y_0=\psi_h(x_0)$: one
linear solve per step, no Newton iteration.

**Mechanical lifts (lemma).** For $H=\tfrac12p^\top M^{-1}p+V(q)$ with $V$ quartic and
$y_{ij}=q_iq_j$, all constraints are protected, and with $v=M^{-1}p$

$$y=qq^\top-\frac{h^2}{4}\,vv^\top\qquad(\text{plus a constant matrix that is zero if you start on the shifted sheet}).$$

For the quartic oscillator, substituting this into Kahan's equations gives the scheme on $(q,p)$

$$q'=q+\tfrac h2(p+p'),\qquad p'=p-\tfrac h2\,qq'(q+q')+\tfrac{h^3}{8}\big(p'^2q+p^2q'\big).$$

This is the "projected Kahan" map $\psi_h$ of the first note, written out. It is *nonlinearly* implicit in $(q,p)$ (note the $p'^2$), yet the lifted form computes it with
one linear solve. Check: the two agree to $4.3\times10^{-13}$ over 2000 steps. For FPU-type
lattices the same works with $r_j=d_j^2-\tfrac{h^2}4\dot d_j^2$, where $d_j=q_{j+1}-q_j$. So the
lifted Kahan method for these lattices is a genuine one-step method on $(q,p)$, even though the
energy itself is not protected.

**Trigonometric lifts.** For the pendulum the circle is protected:
$(s^2+c^2)/(1+\tfrac14h^2p^2)$ is conserved. Starting with $s_0^2+c_0^2=1+\tfrac14h^2p_0^2$ means
$(s,c)=\sqrt{1+\tfrac14h^2p^2}\,(\sin\theta,\cos\theta)$, a one-step method on $(\theta,p)$.

**One-DOF family (§3).** $c_h=0$ is a quadratic equation in $y$ whose coefficients depend on $x$,
because the denominators $D_\lambda$ depend on $x$ only. Take the root that tends to $\varphi(x)$
as $h\to0$.

---

## 5. A catalogue of Kahan-friendly classes

| Class | Lifted variables | What Kahan conserves | Mechanism | Check |
|---|---|---|---|---|
| quartic potentials in positions (FPU-$\beta$, $\varphi^4$ lattices) | $y_{ij}=q_iq_j$ or $r_j=d_j^2$ | every constraint, $y-qq^\top+\tfrac{h^2}4vv^\top$; the energy only in special cases | lemma | FPU, 3 masses: 4 constraints protected, energy not |
| 1-DOF $U(\varphi)+\kappa$ | $y=\varphi(x)$ | constraint (with partner), energy, measure: integrable | pencil + CMOQ Thm 2 | §3 |
| single-harmonic trigonometric potentials with constant mass matrix, $H=\tfrac12\pi^\top G\pi-\sum_kg_k\cos\theta_k$ | $(s_k,c_k)=(\sin\theta_k,\cos\theta_k)$ | circles $(s_k^2+c_k^2)/(1+\tfrac14h^2\dot\theta_k^2)$ and energy $E-\tfrac{h^2}8\dot\pi^\top G\dot\pi$ | two variables + lemma | coupled pendula, $G$ non-diagonal: drift $\le3.7\times10^{-14}$ over $2\times10^5$ steps; true energy error bounded at $1.5\times10^{-3}$ |
| exponential (Toda-type) potentials $\sum_kc_ke^{\alpha_k\cdot q}$ | $u_k=e^{\alpha_k\cdot q}$ only (no constraints at all) | energy $E-\tfrac{h^2}8\dot p^\top M^{-1}\dot p$ | lemma ($\dot p$ is *linear* in $u$) | open Toda, 5 particles: drift $4.4\times10^{-15}$ over $10^5$ steps (plain energy error $1.2\times10^{-2}$) |
| Morse $V=D(1-e^{-a(r-r_0)})^2$ | $u=e^{-a(r-r_0)}$ | energy (two variables $(p,u)$, $A=au$) | CMOQ | drift $1.2\times10^{-14}$ over $10^5$ steps |
| generalised Suslov in $\mathbb R^{2m}$ | none | every normal-mode action | block theorem | $\mathbb R^4$: $7.7\times10^{-15}$ and $5.6\times10^{-15}$ |
| tops: BH dimer in Bloch variables, $H=-2JX-Z^2$ | none | Casimir and energy: integrable | lemma + pencil | orbits on two quadrics |

Two design rules come out of this table:

1. **Lift only with products of variables that move affinely.** In mechanical systems these are
    positions. Then every constraint is protected.
2. **Prefer nonlinearities whose force is linear in the lifted variables**, such as exponentials
    and single harmonics. Then the energy is protected as well. Quartic potentials fail this rule
    (the force $-qy$ is quadratic), which is why their energies are protected only in one DOF.

The Toda and BH-dimer cases may already be known in the integrable-systems literature (Petrera,
Pfadler and Suris survey many such Kahan–Hirota–Kimura discretisations). I did not find these
specific invariants in a quick search, but I did not search exhaustively either.

---

## 6. Invariant measures, and an area form on the shifted sheet

A measure $dz/P(z)$ is preserved when $P(z')=\det(D\Phi(z))\,P(z)$. In other words, $P$ is a
discrete Darboux polynomial whose cofactor is the Jacobian determinant. This is the method of
Celledoni, Evripidou, McLaren, Owren, Quispel and Tapley (2019), which I used for exact searches
over the rationals.

| System | Preserved measure | Source / check |
|---|---|---|
| quartic oscillator | $dz/(1+\tfrac12h^2q^2)$ | first note |
| pendulum | $dz/(1+\tfrac14h^2p^2)$ | first note |
| 1-DOF family (§3) | $dz/(D_{\lambda_1}D_{\lambda_2})$, the product of the CMOQ denominators | CMOQ Thm 2; defect $\approx8\times10^{-16}$ |
| mixed example | $dz/\big((1+\tfrac12h^2u^2)(1+\tfrac12h^2v^2)\big)$; either factor alone fails (defect $8\times10^{-3}$) | `measure_check.py` |
| quartic $V$, lift $y=q^2$ (four tested, including linear and cubic terms) | $dz/P$ with $P$ a polynomial of degree 4, unique | **new**; exact at $h=\tfrac15$ |
| $V=\tfrac14q^4+\tfrac13q^3+\tfrac12q^2$, **all** $h$ | $P$ below | **new**; exact with $h$ symbolic |

For the non-symmetric example ($t=h^2$):

$$P=432+72t\,(1+2q+2q^2+y)+t^2\big(-9(y-2q^2)^2+12q(y-2q^2)+16q^2-10y-12q-9\big)+t^3p^2(1+3q)^2 .$$

On the sheet $y=q^2$ the first correction is $\tfrac{h^2}{6}V''(q)$. The general quartics have the
same shape; for example, their $p^2$-term is always proportional to $V'''(q)^2$. The natural guess
of a single CMOQ-type denominator, $1+\tfrac{h^2}2q(q+\tfrac13)$, fails here (defect
$1.4\times10^{-3}$).

**Why this matters: an area form, hence bounded energy.** A 3D map that preserves a measure
$\mu$ and a function $c_h$ also preserves, on each level set of $c_h$, the 2D area form $\omega$
defined by $\mu=dc_h\wedge\omega$. The back-substituted method of §4 lives on the level set
$c_h=0$. So **for every quartic 1-DOF potential tested, the back-substituted Kahan method
preserves an area form on the original phase space**: it is symplectic for a modified area form.
It is also a near-identity map, and for a nonlinear oscillator (frequency changes with amplitude)
KAM theory (Moser's twist theorem) gives invariant curves that trap orbits.
This explains why the energy error stayed bounded for the non-symmetric quartic in the first note,
even though no modified energy exists.

**Conjecture (supported by exact searches).** If $C$ is a two-variable quadratic and $H$ is *any*
quadratic on $\mathbb R^3$, Kahan's map of the Nambu field $\nabla H\times\nabla C$ preserves
$dz/P$ with $P$ a polynomial of degree 4. Evidence:

* exact solutions found for three random $H$ (with $C=y-q^2$) at $h=\tfrac15$ and $h=\tfrac34$;
* exact solutions for four quartic potentials;
* an all-$h$ formula for one example.

CMMOQ 2014 prove the homogeneous case and CMOQ 2019 the case where *both* integrals are
two-variable. The general case would cover every MQAV lift of a quartic 1-DOF potential.

---

## 7. A recipe for designing Kahan-friendly lifts

1. Write the lifted quadratic field $F$ and list the constraints you need.
2. Compute the space of quadratic integrals. The condition $\nabla I\cdot F\equiv0$ is linear in
    the coefficients of $I$.
3. **Lemma members:** integrals with $S\,Q=0$ (also linear).
4. **Two-variable members:** combinations with $\operatorname{rank}[\,S\mid s\,]\le2$. With two
    integrals this is a cubic equation in the pencil parameter; with three it is an intersection
    of two plane cubics (a resultant); with more, search pencils between pairs (a heuristic that
    can miss members). Keep the real and imaginary parts of complex members.
5. **Block members:** look for blocks that move along a fixed linear flow at one affine speed
    (by hand for now).
6. Form the protected span $\mathcal P$ and check whether every constraint, and the energy, lies
    in it. If not, change the lift (design rules in §5), or accept slow triangular drift (§2.3).

Steps 2–4 and 6 are implemented in `protected_span.py`. Results:

| Lifted system | Quadratic integrals | Lemma | Two-variable | $\mathcal P$ | Constraints | Energy |
|---|---|---|---|---|---|---|
| quartic oscillator $(q,p,y)$ | 2 | 1 | 2 | 2 | protected | protected |
| non-symmetric quartic | 2 | 1 | 1 | 1 | protected | **not** |
| mixed, $y=qp$ | 2 | 0 | 2 | 2 | protected (partner) | protected |
| paper's example, reduced lift $(q^2,p^2,qp)$ | 2 | 0 | 1 | 1 | Casimir **not** | protected |
| paper's example, Lie–Poisson lift | 2 | 0 | 1 | 1 | not exact integrals | not applicable |
| pendulum $(s,c,p)$ | 2 | 1 | 2 | 2 | circle protected | protected |
| octic chain $(q,p,y_1,y_2,u)$ | 3 | 1 | 1 | 2 | $y_1-q^2$ protected; $u-qy_1$ and $y_2-y_1^2$ not exact | protected |
| FPU-$\beta$, 3 masses $(q,p,r)$ | 5 | 4 | 0 (heuristic missed them) | 4 | all 4 protected | **not** |
| BH dimer (Bloch) | 2 | 1 | 2 | 2 | Casimir protected | protected |
| coupled pendula, constant mass matrix | 3 | 1 | 2 | 3 | both circles protected | protected |

The same test on the resonant block of §1.3 reports $N=\lvert v\rvert^2$ protected and
$G=u^\top\mathbb Jv$ "integral, NOT protected": mechanisms A and B miss it, and only the block
theorem protects it.

---

## 8. What is known and what is new

**Known (and used here).**

* CMOQ 2019 Theorem 1, two-variable integrals: also the idea of applying it after linear changes
    of variables.
* CMOQ 2019 Theorem 2, the integrable Nambu family with product measure: §3 is an instance.
* CMMOQ 2014: homogeneous quadratic Nambu systems (measure and integrability), generalised Suslov
    systems (modified energy and measure), Ishii-type invariants (now special cases of the lemma).
* Celledoni–McLachlan–Owren–Quispel 2013: modified energy and measure for cubic Hamiltonians with
    constant Poisson structure.
* Petrera–Pfadler–Suris 2011: survey of integrable Kahan–Hirota–Kimura maps.
* Celledoni–Evripidou–McLaren–Owren–Quispel–Tapley 2019: discrete Darboux polynomials.
* Mañosa–Pantazi 2024: Kahan maps that preserve the *original* cubic Hamiltonian.

**New here, as far as I can tell.**

1. The companion lemma in general form ($S\,Q=0$), with energies (Toda-type, trigonometric) as
    well as constraints.
2. The **block theorem** (single and several speeds), which contains the lemma and CMOQ
    Theorem 1, and gives new modified invariants for non-diagonalisable blocks (squared
    denominators), plus all normal-mode actions of generalised Suslov systems.
3. The **protected span** and an automatic test of which constraints a lift protects.
4. The **classification** of exact product constraints (squares, affine factors, opposite
    scaling), the partner mechanism, and the degree-regress rule of thumb.
5. The observation that MQAV lifts of $U(\varphi)+\kappa$ are CMOQ Nambu systems after the pencil
    change of variables.
6. **Back-substitution:** the lifted Kahan method as a linearly implicit implementation of a
    symmetric one-step method on the original variables.
7. **Degree-4 invariant densities** for the lifted quartic 1-DOF Kahan maps tested (closed form
    for all $h$ in an example), the resulting area form on the shifted sheet, and the Nambu
    conjecture.
8. Kahan-friendly classes: Toda-type potentials in exponential variables, single-harmonic
    trigonometric potentials with a constant mass matrix, and triangular hierarchies.

---

## 9. Open questions (and which ones look publishable)

1. **Prove the measure conjecture of §6.** *(Solved for every quartic potential: the paper's Proposition 5.3
    gives the density in closed form, see [`paper/proofs/quartic_density_proof.py`](paper/proofs/quartic_density_proof.py).)*
    This would give a clean theorem: "Kahan's method on the
    quadratic lift of any quartic 1-DOF Hamiltonian is either integrable or area-preserving on
    the shifted sheet." Combined with §3 and §4, that is a self-contained paper. The structure of
    $P$ (the $\tfrac{h^2}{6}V''$ term and the $V'''^2$ term) suggests a closed formula.
2. **Non-commuting speeds.** Opposite scaling is the simplest non-commuting case. Partners rescue
    it in §3. Is there a general statement, for example in terms of the Lie algebra generated by
    the $L_k$?
3. **Several DOF.** Is there a lift of FPU-$\beta$ whose energy is protected too? Or does the
    lifted Kahan map at least preserve a modified Poisson structure on the shifted sheet (the
    analogue of §6)? A measure search for two masses is feasible.
4. **Higher order.** Modified invariants depend on $h$, so compositions of Kahan steps with
    different step sizes conserve none of them exactly. Is there a composition that keeps a
    common invariant, for example by choosing the lift per stage?
5. **PDEs.** Exponential nonlinearities (Toda lattices, Liouville-type equations) and
    single-harmonic ones (XY chains) are the natural protected PDE classes. Trigonometric lifts
    that keep $q$ (sine-Gordon) suffer phase drift.

---

## 10. Reproducibility

All scripts are in [`code-invariants/`](code-invariants/); its README maps each claim above to a
script. They need Python 3 with `numpy`, `scipy` and `sympy`, and run in seconds to a few
minutes.

## References

* B. K. Tapley, Geometric integration of ODEs using multiple quadratic auxiliary variables, *SIAM J. Sci. Comput.* 44(4) (2022) A2651–A2668, [arXiv:2108.06548](https://arxiv.org/abs/2108.06548).
* E. Celledoni, D. I. McLaren, B. Owren, G. R. W. Quispel, Geometric and integrability properties of Kahan's method: the preservation of certain quadratic integrals, *J. Phys. A* 52 (2019) 065201, [arXiv:1805.08382](https://arxiv.org/abs/1805.08382).
* E. Celledoni, R. I. McLachlan, D. I. McLaren, B. Owren, G. R. W. Quispel, Integrability properties of Kahan's method, *J. Phys. A* 47 (2014) 365202, [arXiv:1405.3740](https://arxiv.org/abs/1405.3740).
* E. Celledoni, R. I. McLachlan, B. Owren, G. R. W. Quispel, Geometric properties of Kahan's method, *J. Phys. A* 46 (2013) 025201, [arXiv:1209.1164](https://arxiv.org/abs/1209.1164).
* E. Celledoni, C. Evripidou, D. I. McLaren, B. Owren, G. R. W. Quispel, B. K. Tapley, Detecting and determining preserved measures and integrals of birational maps, [arXiv:1902.04685](https://arxiv.org/abs/1902.04685).
* M. Petrera, A. Pfadler, Yu. B. Suris, On integrability of Hirota–Kimura type discretizations, *Regul. Chaotic Dyn.* 16 (2011) 245–289, [arXiv:1008.1040](https://arxiv.org/abs/1008.1040).
* V. Mañosa, C. Pantazi, Kahan–Hirota–Kimura maps preserving original cubic Hamiltonians, [arXiv:2405.01321](https://arxiv.org/abs/2405.01321).
* W. Kahan, Unconventional numerical methods for trajectory calculations, unpublished lecture notes (1993).
