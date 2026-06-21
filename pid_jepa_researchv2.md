## 5. PID-JEPA: deriving the controller for the embedding distribution

### 5.1 The plant (controlled output = distribution covariance over depth)

Let \(Z_\ell \in \mathbb{R}^{B\times d}\) be the batch of per-sample embeddings at depth
\(\ell\) (e.g. pooled token or `[CLS]`), \(\tilde Z_\ell = Z_\ell - \bar Z_\ell\), and

$$
\Sigma_\ell = \tfrac{1}{B-1}\tilde Z_\ell^\top \tilde Z_\ell .
$$

From §1.2, the uncontrolled across-sample dynamics over depth are dissipative; per
eigen-mode \(k\) with rate \(a_k>0\):

$$
\dot\lambda_k = -2 a_k \lambda_k \quad(\text{open loop, collapses}).
$$

The controller adds \(c_k\): \(\;\dot\lambda_k = -2 a_k \lambda_k + c_k.\)

### 5.2 The reference \(v_0\) (non-heuristic — measured from the input)

Following PIDformer's "follow the initial value", choose the reference as the **isotropic
redistribution of the input's energy**:

$$
\boxed{\;v_0 \;\equiv\; R \;=\; \sigma_0^2 I, \qquad \sigma_0^2 = \frac{\mathrm{tr}(\Sigma_0)}{d}\;}
$$

where \(\Sigma_0\) is the covariance of the **layer-0 (patch/input) embeddings**, which is
full-rank because inputs are diverse. This is **not a tuned hyperparameter**: \(\sigma_0^2\)
is *measured from the data*. It preserves the total signal energy of the input while
spreading it isotropically so no direction can degenerate. Error per mode:

$$
e_k = \sigma_0^2 - \lambda_k .
$$

At collapse \(\lambda_k \to 0 \Rightarrow e_k \to \sigma_0^2 \neq 0\): a **persistent
error**, exactly the condition under which integral action is decisive.

### 5.3 P, I, D — derived from the plant

**Proportional.** \(c_k^P = k_P e_k = k_P(\sigma_0^2 - \lambda_k)\). Closed loop:

$$
\dot\lambda_k = -2a_k\lambda_k + k_P(\sigma_0^2 - \lambda_k)
\;\Rightarrow\;
\boxed{\lambda_k^\star = \frac{k_P}{2a_k + k_P}\,\sigma_0^2 < \sigma_0^2.}
$$

P **prevents collapse** (\(\lambda_k^\star > 0\) for \(k_P>0\)) but leaves a **steady-state
offset** \(e_k^\star = \frac{2a_k}{2a_k+k_P}\sigma_0^2 > 0\): larger \(k_P\) shrinks but
never removes it.

**Integral.** Add state \(w_k\), \(\dot w_k = e_k\):

$$
\dot\lambda_k = -2a_k\lambda_k + k_P e_k + k_I w_k, \qquad \dot w_k = \sigma_0^2 - \lambda_k.
$$

At steady state \(\dot w_k = 0 \Rightarrow e_k = 0\):

$$
\boxed{\lambda_k^\star = \sigma_0^2 \ \text{(zero steady-state error)}, \qquad
w_k^\star = \frac{2a_k \sigma_0^2}{k_I} \neq 0.}
$$

The integrator settles at a **non-zero** value \(w_k^\star\) that supplies a **persistent
restoring force** \(k_I w_k^\star = 2a_k\sigma_0^2\) — exactly balancing dissipation. This
is the Internal Model Principle in action and the rigorous justification for the I term.

**Derivative (honest treatment).** \(c_k^D = k_D \dot e_k = -k_D \dot\lambda_k\). The
\((\lambda_k, w_k)\) system has characteristic polynomial

$$
(1+k_D)\,s^2 + (2a_k + k_P)\,s + k_I = 0,
\quad
\omega_n = \sqrt{\tfrac{k_I}{1+k_D}}, \quad
\zeta = \frac{2a_k + k_P}{2\sqrt{k_I (1+k_D)}}.
$$

Two consequences, stated precisely:
- The steady state is **unchanged** by \(k_D\) (since \(\dot e = 0\) there) — D never harms
  the target.
- In this **first-order plant**, \(k_D\) acts as added inertia and *reduces* \(\zeta\), so
  **D does not damp overshoot in the idealized model** (matching our static-probe
  experiments where D had no effect). D's genuine benefit is **phase lead in the realistic
  higher-order/delayed plant** (an \(L\)-layer encoder with actuator lag): there it
  improves the stability margin and suppresses the oscillatory windup observed in PI-only
  runs. **Recommendation:** keep D as a principled, optional lead compensator, not as a
  claimed damper of the first-order loop.

**Full PID-JEPA control law (per mode, then lifted to matrices):**

$$
\boxed{\;c_k = k_P(\sigma_0^2-\lambda_k) + k_I\!\!\int_0^t (\sigma_0^2-\lambda_k)\,ds + k_D \frac{d}{dt}(\sigma_0^2 - \lambda_k)\;}
$$

### 5.4 Stability of the closed loop

The Jacobian of \((\lambda_k, w_k)\) at \((\sigma_0^2, w_k^\star)\) has
\(\mathrm{tr} = -\tfrac{2a_k+k_P}{1+k_D} < 0\) and \(\det = \tfrac{k_I}{1+k_D} > 0\); by
Routh–Hurwitz the equilibrium is **asymptotically stable** for all
\(k_P, k_I > 0,\, k_D > -1\). Matrix Lyapunov function
\(V(\Sigma) = \tfrac12\|\Sigma - \sigma_0^2 I\|_F^2\) decreases along the closed-loop flow
for gains in this range, and the collapsed point \(\Sigma = 0\) yields \(V = \tfrac{d}{2}\sigma_0^4\),
a strict maximum on the PSD ray — i.e. **collapse is unstable, isotropy is the attractor.**

### 5.5 The actuator (turning \(c_k\) into a ViT operation) and the obstruction fix

We need a forward-pass operation whose effect on \(\Sigma_\ell\) equals
\(\eta\, \mathcal{C}(E_\ell)\), where the PID signal in matrix form is

$$
\mathcal{C}(E_\ell) = k_P E_\ell + k_I \!\!\sum_{j\le\ell} E_j + k_D (E_\ell - E_{\ell-1}),
\qquad E_\ell = \sigma_0^2 I - \Sigma_\ell .
$$

Apply an affine correction in the residual stream,
\(Z_{\ell+1} = Z_\ell + \eta\, \tilde Z_\ell M_\ell^\top\), giving to first order
\(\Sigma_{\ell+1} \approx \Sigma_\ell + \eta(M_\ell\Sigma_\ell + \Sigma_\ell M_\ell^\top)\).
Matching to \(\eta\,\mathcal{C}(E_\ell)\) is a **Lyapunov equation**
\(M_\ell\Sigma_\ell + \Sigma_\ell M_\ell^\top = \mathcal{C}(E_\ell)\); in the eigenbasis of
\(\Sigma_\ell\),

$$
m_k = \frac{c_k}{2\lambda_k}.
$$

**Obstruction:** \(m_k \to \infty\) as \(\lambda_k \to 0\) (the §3.3 loss of authority). Fix
with the two escapes of §3.3:
1. **Preventive:** insert the controller after several blocks and at every subsequent block,
   so \(\lambda_k\) never reaches \(0\) (a regularized inverse \((\Sigma_\ell+\epsilon I)^{-1}\)
   suffices while \(\lambda_k \gg \epsilon\)).
2. **Integral-induced instability (the invariance-safe escape).** Exact collapse
   \(\tilde Z_i = 0\) is measure-zero; the integral term grows the deviation gain
   \((I+\eta G_\ell)\) past \(1\) along deficient modes, so any infinitesimal between-image
   difference is *amplified* — collapse becomes a strictly unstable equilibrium. This is the
   **same mechanism VICReg uses** and it needs no per-sample signal.

   *Rejected:* a deterministic per-sample additive code \(Z_i \leftarrow Z_i + \sqrt{k_I w_k}\,h_{i,k}\)
   indexed by batch position is **not** content-derived, so it would assign *different*
   codes to the local/global views of the same image and fight invariance (see §5.7). Only
   use additive injection if \(h\) is a function of shared content — which is what we are
   trying to learn, hence circular. Prefer integral-induced instability.

### 5.6 Unification: existing methods are degenerate cases of this controller

| Method | Controlled stat | P | I | D | Reference | Lyapunov? |
|---|---|---|---|---|---|---|
| BatchNorm | per-dim mean/var | ✓ (dead-beat) | – | – | \(0, 1\) | – |
| DINO centering | 1st moment (mean) | – | ✓ (leaky) | – | \(0\) | – |
| Whitening / CW-RGP | full \(\Sigma\) | ✓ (dead-beat) | – | – | \(I\) | – |
| DirectPred | \(\Sigma\) spectrum | ✓ (feedforward) | – | – | \(\sqrt{\Sigma}\) | – |
| PIDformer | token state | ✓ | ✓ (leaky) | (✓) | \(v_0=f\) (token) | partial |
| **PID-JEPA (this note)** | \(\Sigma\) spectrum (sample) | ✓ | ✓ | ✓ | \(\sigma_0^2 I\) from input | ✓ |
The table is the novelty statement: prior work occupies isolated, incomplete cells; we
provide the **complete, derived, stability-certified controller** on the axis that governs
JEPA collapse.

### 5.7 Invariance-safety: the controller must read batch statistics, never per-sample targets

With \(y=\) image identity, the **law of total covariance** splits the embedding spread:

$$
\underbrace{\mathrm{Cov}(z)}_{\text{marginal (batch)}}
= \underbrace{\mathbb{E}_y[\mathrm{Cov}(z\mid y)]}_{\text{within-image (across views)}}
+ \underbrace{\mathrm{Cov}_y(\mathbb{E}[z\mid y])}_{\text{between-image}} .
$$

- **Invariance** drives the within-image term \(\to 0\) (local & global views → one point).
- **Anti-collapse** needs the between-image term \(\succ 0\) (distinct images → distinct points).
- **Collapse** = both terms \(\to 0\).

A per-sample reference \(r_i=\Phi f(x_i)\) (§6.2) has a target that **depends on the view**
\(x_i\), so it injects energy into the *within-image* term and pulls the two views of one
image toward different anchors — **directly antagonistic to invariance**. Hence per-sample
reference tracking is invalid for multi-view JEPA (it is valid only in PIDformer's
single-input-per-state token setting).

**Consequence.** The controller may read only quantities **invariant to which view a sample
is** — permutation-invariant batch statistics of the marginal (\(\mu_B,\Sigma_B\), or the
batch distribution). The §5.5 covariance actuator applies a *shared* map
\(z_i\leftarrow z_i+\eta G(z_i-\bar z)\) with \(G=\mathrm{PID}(\sigma_0^2I-\Sigma_B)\): for
\(z_a\approx z_b\) (same image) the corrections are \(\approx\) equal, so \(\|z_a-z_b\|\) is
preserved and the invariance optimum \(z_a=z_b\) is a fixed set of the layer. The controller
reshapes the **marginal / between-image** term *only*, leaving invariance untouched. This is
exactly why SIGReg/VICReg regularize the marginal and are orthogonal to invariance.

### 5.8 The distributional setpoint (matching LeJEPA's Epps–Pulley target)

Covariance control fixes only the 2nd moment (VICReg-level). SIGReg matches the full 1-D law
to \(\mathcal N(0,1)\) via the empirical characteristic function (ECF) on random slices \(v\):

$$
\hat\phi_v(t)=\tfrac1B\sum_i e^{\,i t\langle v,z_i\rangle},\qquad
\phi^\star(t)=e^{-t^2/2},\qquad e_v(t)=\phi^\star(t)-\hat\phi_v(t).
$$

The setpoint becomes a **function**, and the control law is a PID on \(e_v(t)\):

$$
c_v(t)=k_P e_v(t)+k_I\!\!\int e_v(t)\,d\tau+k_D\,\partial_\tau e_v(t),
$$

realized as a forward-pass transport reducing the Epps–Pulley statistic
\(\int|e_v(t)|^2 w(t)\,dt\), with integral memory across batches. The **covariance-PID is the
2nd-order truncation**: \(\log\hat\phi_v(t)\approx i t\,v^\top\mu-\tfrac{t^2}{2}v^\top\Sigma v\),
and matching the \(t^2\) term to \(-\tfrac{t^2}{2}\) gives \(v^\top\Sigma v=1\,\forall v
\Rightarrow \Sigma=\sigma_0^2 I\). This yields a hierarchy of **invariance-safe** batch-statistic
controllers — mean (= DINO centering, leaky-I), covariance (our core PID), full CF
(= LeJEPA's target) — climbing from VICReg-level to LeJEPA-level anti-collapse, all as
forward-pass controllers rather than losses.

---

## 6. Concrete architectural instantiations

### 6.1 Spectral PID block (recommended first prototype)

A drop-in module placed after blocks \(\{L/2, \dots, L\}\) of a ViT encoder. Carries
controller state as **buffers** (like BN running stats):
- buffers: \(W \in \mathbb{R}^{d\times d}\) (integral), \(E_{\text{prev}}\) (for D), `step`;
- forward (training and inference): compute \(\Sigma_\ell\) over the batch, \(E=\sigma_0^2 I-\Sigma_\ell\),
  update \(W \leftarrow \mathrm{clip}(W + \eta E)\), form \(\mathcal{C}(E)\), solve
  \(M_\ell\) (regularized), apply \(Z\leftarrow Z + \eta\,\tilde Z M_\ell^\top\) (+ additive
  channel for tiny eigenvalues);
- **no loss term** is added; total objective = invariance (+ optional probe for monitoring).

\(\sigma_0^2 = \mathrm{tr}(\Sigma_0)/d\) is computed from the patch-embedding covariance on
the first batch and frozen (or slow-EMA tracked).

### 6.2 Reference-tracking variant — INVALID for multi-view JEPA (kept for the record)

One could regulate each sample to a fixed low-rank reference \(r_i = \Phi f(x_i)\),
\(\Phi\in\mathbb{R}^{m\times d}\), with PID on \(e_i=r_i-\Phi z_i\) in the residual stream:

$$
Z_{\ell+1} = Z_\ell + \lambda_P\,\Phi^\top e_\ell + \lambda_I\,\Phi^\top\!\!\sum_{j\le\ell}e_j + \lambda_D\,\Phi^\top(e_\ell-e_{\ell-1}).
$$

**This is incompatible with multi-view JEPA** (§5.7): local and global views of the *same*
image have different inputs \(x_{\text{local}},x_{\text{global}}\), so anchoring them to
\(\Phi f(x_{\text{local}})\) and \(\Phi f(x_{\text{global}})\) inflates the *within-image*
covariance term and forces the two views apart — destroying the invariance objective. The
target depends on the view, which is exactly the property a JEPA controller must **not** have.

It remains valid only where there is a single input per controlled state (PIDformer's
token-level oversmoothing). For sample-level JEPA, use the **batch-statistics** controller of
§6.1 (covariance) or §5.8 (full distribution), whose setpoint is a view-agnostic marginal
statistic and is therefore invariance-safe.

### 6.3 Why isotropy must be checked with `erank`

By §3.3, \(\Sigma\approx\sigma_0^2 I\) can be achieved vacuously. The honest validation is
**effective rank of the embedding** and the **linear-probe accuracy with the anti-collapse
loss removed**. If `erank` is high *and* probe accuracy holds with no loss-side
regularizer, the architecture is genuinely preventing collapse.

---

## 7. Positioning / novelty vs the literature

1. **Mechanism, not penalty.** Unlike VICReg/SIGReg/LeJEPA/VJ-VCR/C-JEPA, anti-collapse is
   in the **plant** (forward pass), active at inference, optimizer-independent.
2. **Derived, not heuristic.** Unlike EMA+stop-grad, DINO centering, T-JEPA reg-tokens, all
   gains follow from the plant dynamics; the reference \(\sigma_0^2 I\) is *measured*.
3. **Complete controller with guarantees.** Unlike whitening/CW-RGP (dead-beat, memoryless)
   and DirectPred (feedforward), we add **integral action** (zero steady-state error via
   Internal Model Principle) and a **Lyapunov/Routh–Hurwitz certificate**.
4. **Right axis.** Unlike PIDformer (token oversmoothing), we regulate the **sample/feature
   distribution** that governs JEPA collapse, and we identify and resolve the
   **controllability obstruction** that defeats naive output normalization.
5. **A unifying lens.** BN, DINO-centering, whitening, DirectPred, PIDformer become
   degenerate cells of one controller table (§5.6).

One-sentence thesis: *Representation collapse is the attracting consensus mode of the
encoder's dissipative dynamics; PID-JEPA embeds a derived PID controller in the ViT forward
pass that renders collapse unstable and isotropy the zero-steady-state attractor — the
distributional guarantee of LeJEPA, achieved by architecture instead of a loss.*
