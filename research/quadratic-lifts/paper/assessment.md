# How strong is the draft, and what should happen before we submit?

> **Main takeaway.** The draft is a solid, publishable paper. For *Nonlinearity* I would rate it as a
> plausible but not safe submission, most likely to come back with "major revision". For *J. Phys. A*,
> where most Kahan-method papers appear, it is a strong fit. The reasons:
>
> * Its three best results are clean, proved and new. These are the block theorem, the Liouville-measure
>   theorem and an integrable family in every dimension.
> * Its broadest message is still a conjecture backed by experiments. That message is which integrable
>   systems survive lifting.
> * One corollary rests on an unpublished companion manuscript.

## 0. Terms and notation used below

| Term | Meaning |
|---|---|
| ODE | ordinary differential equation, $\dot z=F(z)$ |
| DOF | degree of freedom (one position–momentum pair $(q,p)$) |
| $h$, $\Phi_h$ | time step; one step of Kahan's method, $z\mapsto z'$ |
| Kahan's method | for a quadratic $F$: $\frac{z'-z}{h}=Q(z,z')+\frac12B(z+z')+b$; costs one linear solve per step |
| quadratic lift | extra variables, e.g. $y=q^2$, that turn a non-quadratic ODE into a quadratic one in more variables |
| modified integral $I_h$ | a function $I+O(h^2)$ that Kahan's map conserves **exactly**, step after step |
| integrable map | a map with so many conserved quantities (plus a preserved volume) that every orbit lies on a small surface, here a 2-torus; no chaos |
| degree growth $d_k$ | how fast the algebraic degree of the $k$-th iterate grows; polynomial growth signals integrability, exponential growth signals chaos |
| [LP] | the uploaded draft "Kahan's method for Lie–Poisson systems: orbit hulls, measures and integrability" |

## 1. What the paper proves, in plain words

1. **Block theorem (Theorem 3.2).** Take a group of variables that is pushed along *fixed* directions, with a
   "speed" that depends on the rest of the system. Then Kahan's method keeps a slightly rescaled copy of each
   quadratic invariant of those directions. This holds even when the directions do not commute.
   *Example:* for a particle in a round (isotropic) well, $q\mapsto$ position, $p\mapsto$ momentum, Kahan keeps
   the angular momentum $M=q_1p_2-q_2p_1$ in the form $M/(1+\frac12h^2U'(y))$ exactly. The theorem also
   contains a known result (Celledoni–McLaren–Owren–Quispel 2019) as a special case.
2. **Lift constraints (Theorems 4.2, 4.4).** If the factors of a constraint move at a constant or linear rate,
   Kahan keeps a modified constraint exactly.
   *Example:* the quartic oscillator with $y=q^2$ keeps $y-q^2+\frac{h^2}{4}p^2$. Starting on
   $y=q^2-\frac{h^2}{4}p^2$, the lifted method never drifts. It is then an honest one-step method on $(q,p)$.
3. **Liouville measure (Theorem 5.1).** For any potential made of exponentials and cosines (Toda chains, rotor
   chains, coupled pendula), with any mass matrix and in any dimension, the lifted Kahan map preserves phase-space
   volume exactly. The proof is one page of elementary algebra: three "shears".
4. **An integrable family in every dimension (Theorem 6.1, Corollary 6.2).** For
   $H=\frac12\lvert p\rvert^2+\frac12u_2\lvert q\rvert^4+u_1\lvert q\rvert^2$ in $\mathbb R^n$, the lifted Kahan
   map keeps:
   * a modified constraint;
   * all modified angular momenta;
   * a modified energy;
   * a measure.

   Every regular orbit lies on an invariant 2-torus. **All of this now has short proofs by hand.** While
   drafting I found two arguments:
   * the modified energy follows from a "crossed" version of the block theorem plus the discrete angular
     momentum $q_k\wedge q_{k+1}=hM_h$;
   * the measure follows from two scalar relations that Kahan's equations imply.

   Computer algebra now serves as an independent check, with $h,u_1,u_2$ symbolic and $n=2,3$, in about 8
   minutes.

   *Practical bonus:* the step has a closed form with cost $O(n)$, like Störmer–Verlet. It stays stable at
   $h\omega=10$, where Störmer–Verlet already fails at $h\omega>2$.
5. **Survey (Section 7).** Nineteen lifted maps were tested by exact degree growth.
   * Systems that are integrable through a linear symmetry plus one DOF stay integrable after lifting.
   * Systems integrable through hidden symmetries do not: three-particle Toda, Garnier, and the quartic
     $1{:}6{:}8$ and $1{:}12{:}16$ cases.
   * *Example:* the same potential $q_1^4+6q_1^2q_2^2+q_2^4$ gives an integrable map with the lift
     $y_\pm=(q_1\pm q_2)^2$ and a chaotic map with $y_k=q_k^2$.

## 2. Scorecard against what referees ask of a paper on integrable maps

| Criterion | Rating | Why |
|---|---|---|
| Is the main map new? | medium | Integrable discretisations of central-force and quartic problems exist by other methods (Feng–Maruno 2008; discrete Newton equations in Suris 2003). Ours is new *as a Kahan map with closed-form invariants in every dimension*, but no conjugacy check against the known maps has been done. |
| Is integrability proved, and is the structure explicit? | strong | All integrals and the measure are explicit and proved by hand (with an independent computer-algebra check). The tori theorem is proved. |
| Family or mechanism, not one example? | strong | The block theorem and the Liouville theorem are mechanisms. The isotropic family covers every $n$ and every quadratic $U$. |
| Natural origin? | strong | A standard integrator applied to classical mechanical systems. |
| Connections to other structures? | weak–medium | No Lax pair, no explicit (elliptic-function) solution, no Poisson structure for the isotropic map. |
| Interest beyond integrable systems | medium | Explicit-cost, stable, structure-preserving integrators; volume-preserving integrators for Hybrid Monte Carlo (untested). |
| Depth of the main theorem | medium | Each proof is short and elementary. That is good for readers, but some referees equate depth with length. |

## 3. What a referee would most likely object to (most serious first)

1. **Overlap and priority with [LP].** [LP]'s midpoint theorem covers the single-speed block theorem and the
   commuting measure results, and our Corollary 5.3 uses its Theorem 9(iii)(b). That theorem is itself based on
   unpublished computations. If [LP] is yours, merge or submit together. If not, coordinate with its authors
   before submitting.
2. **"Why these formulas?"** This was the main weakness, and it is now largely fixed: the energy and the
   measure have conceptual hand proofs. A referee may still ask for a general reduction theory for Kahan maps
   with symmetry, which the paper only conjectures (Conjecture 7.1).
3. **Section 7 is empirical.** The organising principle (Conjecture 7.1) is not proved even in the easy
   direction beyond the examples.
4. **Novelty of the integrable map.** It may be conjugate to a known integrable map; the check has not been
   done.
5. **The numerical gain is modest.** At equal step size the lifted Kahan map is about 3 times less accurate in
   energy than the implicit midpoint rule. Its advantages are cost (explicit, $O(n)$), stability and exact
   modified invariants.

## 4. Where to send it

* ***Nonlinearity***: worth trying if items 1–2 of §3 are addressed; otherwise expect a demand for more
  depth. *Nonlinearity* likes a strong dynamical statement. The tori theorem and the principle "linear symmetry
  plus one DOF" are what would sell it there.
* ***J. Phys. A: Math. Theor.***: the most natural home. Most Kahan papers cited in the draft appeared there.
  It is a good fit as it stands, after the [LP] question is settled.
* ***Proc. R. Soc. A*** or ***Open Communications in Nonlinear Mathematical Physics***: also natural.
* ***BIT*** or ***IMA J. Numer. Anal.***: only if the paper is rewritten around the integrators. It would need
  comparisons with SAV/IEQ schemes, polarised multistep Kahan and the midpoint rule on lattices.

## 5. What would make it clearly strong (best value first)

1. **Settle the [LP] relationship.** One merged paper, with the midpoint theorem, the block theorem, the
   orbit-hull classification and the isotropic family, would be markedly stronger than either alone.
2. **Generalise the new hand proofs.** The energy proof is a "block theorem without closure", repaired by a
   conserved quantity (the discrete angular momentum). Stating that as a general theorem would explain the
   isotropic family and possibly the magnetic case, and would be the paper's strongest new idea.
3. **Prove one direction of Conjecture 7.1** for the class "rotations plus the one-DOF pencil family". This
   would turn Section 7 from a survey into a theorem.
4. **Conjugacy and literature check** against Feng–Maruno (2008) and discrete Newtonian central-force maps,
   using degree sequences and singularity patterns.
5. **Optional extras:**
   * an explicit solution in elliptic functions;
   * an explanation of the magnetic case;
   * a Hybrid Monte Carlo test for the Toda or rotor integrators.

## 6. Status of the files and the new repository

* The draft is in `research/quadratic-lifts/paper/` (`main.tex`, `main.pdf`, 18 pages), with the proof scripts,
  the figure scripts and a README that maps every result to a script.
* **New GitHub repository:** creating it from this session failed. The GitHub integration returned
  `403 Resource not accessible by integration`, which means it may not create repositories on your account. To
  finish:
  1. create an empty repository on GitHub, e.g. `bentaps/kahan-quadratic-lifts` (private is fine, no README);
  2. make sure the Claude GitHub app can access it (github.com/apps/claude → Configure → repository access);
  3. tell me the name. I will attach it and push the prepared branch, which holds the whole
     `research/quadratic-lifts` folder with its history, as `main`.

## Sources

* IOP Publishing, LaTeX guidance for its journals (class file optional; any common TeX variant accepted):
  [publishingsupport.iopscience.iop.org/questions/latex-template](https://publishingsupport.iopscience.iop.org/questions/latex-template/)
* Feng & Maruno, Integrable discretizations of a two-dimensional Hamiltonian system with a quartic potential:
  [arXiv:nlin/0703027](https://arxiv.org/abs/nlin/0703027)
* Petrera, Suris & Zander, How one can repair non-integrable Kahan discretizations:
  [arXiv:2003.12596](https://arxiv.org/abs/2003.12596)
* Celledoni, McLachlan, McLaren, Owren & Quispel, Discretization of polynomial vector fields by polarization:
  [arXiv:1506.03855](https://arxiv.org/abs/1506.03855)
* Gubbiotti, McLaren & Quispel, Modified Hamiltonians and measures of 2D Kahan maps:
  [arXiv:2309.00799](https://arxiv.org/abs/2309.00799)
* Hone & Quispel, Analogues of Kahan's method for higher order equations of higher degree:
  [arXiv:1911.03161](https://arxiv.org/abs/1911.03161)
* Drozdov, Gubbiotti & Latini, Symmetry algebras of continuous and discrete isotropic oscillators:
  [arXiv:2505.15672](https://arxiv.org/abs/2505.15672)
