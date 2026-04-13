# Research blueprint: Dynamics-grounded tube regularization (beyond cosine STP / TS)

## One-sentence pitch

Replace purely geometric cosine heuristics with a **discrete-time Lyapunov-flavored transversal energy** plus optional **paper-faithful local curvature**, giving a testable stability story (Wiggins Ch.1–3) and ablations that isolate each mechanism.

## Problem with cosine-only baselines

- **STP:** $\(1-\cos\)$ between chord vectors is a **directional** alignment proxy; it does not encode **monotone contraction** of “noise” along time or a **set** (tube) in state space.
- **TS:** $\(1-\cos(v_t,v_{t+1})\)$ is **local**; it does not constrain **global** deviation from a long-horizon secant or Q/A-skipped semantics.

## Formal object (discrete time)

- Hidden states $\(h_t \in \mathbb{R}^D\)$ along token or frame index $\(t\)$.
- **Reference direction** $\(v_{\mathrm{geo}}\)$: secant between semantic anchors (STP with gap) or first–last valid frame (WM).
- **Stitched displacements** $\(d_t\)$ (STP mode) or $\(d_t = h_t - h_{s}\)$ (WM mode) as in `LyapunovTubeLoss`.
- **Transversal component** $\(e_t = d_t - \Pi_{v} d_t\)$, **Lyapunov candidate** $\(V_t = \|e_t\|^2\)$.
- **Training surrogate:** penalize violations of $\(V_{t+1} \le \gamma V_t + \tau\)$ on **valid** transitions (softplus/ReLU).

**Claim style (theorem sketch, not automatic truth):** If the learned map along $\(t\)$ were a contraction in transversal coordinates with slack \($\tau\$), then tube width would be bounded (discrete-time ISS flavor). Training **encourages** this; it does **not** prove stability of inference unless you add explicit model class assumptions.

## Theory correspondence (Stephen Wiggins, *Introduction to Applied Nonlinear Dynamical Systems and Chaos*, 2nd ed.)

This section maps `dynamics_tube_loss.py` to the book’s **Ch.1** (equilibria, stability, **maps**), **Ch.2** (**Liapunov functions**), and **Ch.3** (**invariant manifolds**). Wiggins uses the spelling **Liapunov**.

### Ch.1 — Discrete time as a map

Autoregressive depth index $\(t\)$ is **discrete time**. The model induces

$$
h_{t+1} = F_t(h_t, x_{t+1}; \theta)
$$

(with $\(x_{t+1}\)$ given by data at train time). This matches **§1.3 Maps**: stability and asymptotic behavior are naturally stated for **difference equations**, not only $\(\dot x = f(x)\)$.

**In code:** `h` has shape `(B, T, D)`; all losses scan the time dimension `T`.

### Ch.3 (used first geometrically) — Nominal direction and “tube”

Fix a **reference secant** $\(v_{\mathrm{geo}} \in \mathbb{R}^D\)$ and unit vector $\(\hat v = v_{\mathrm{geo}} / \|v_{\mathrm{geo}}\|\)$.

**World-model mode** (mask-valid endpoints $\(t_{\mathrm{start}}, t_{\mathrm{end}}\))$:

$$
v_{\mathrm{geo}} = h_{t_{\mathrm{end}}} - h_{t_{\mathrm{start}}}, \qquad
d_t = h_t - h_{t_{\mathrm{start}}}.
$$

**STP / Q–A gap mode** (inclusive token bounds $\([u_s,u_e]\)$, $\([a_s,a_e]\))$:

$$
v_{\text{user}} = h_{u_e} - h_{u_s}
$$

$$
d_t =
\begin{cases}
h_t - h_{u_s}, & t \in [u_s, u_e] \\
v_{\text{user}} + (h_t - h_{a_s}), & t \in [a_s, a_e]
\end{cases}
$$

$$
v_{\text{geo}} = v_{\text{user}} + (h_{a_e} - h_{a_s})
$$

**Orthogonal decomposition** (normal coordinate to the reference line in \(d\)-space):

$$
p_t = \langle d_t, \hat v\rangle \,\hat v, \qquad
e_t = d_t - p_t, \qquad e_t \perp \hat v.
$$

The set $\(\{ t : e_t = 0\}\)$ (ideal) lies in an **affine subspace** aligned with $\(\hat v\)$: a **linear surrogate** for an **invariant cylinder / tube** around a nominal direction (Ch.3 *spirit*: structured invariant set; not a full stable-manifold construction).

**In code:** `v_geodesic`, `d_all`, `v_norm`, `e_all` in `LyapunovTubeLoss`.

### Ch.2 — Liapunov candidate on the transversal error

Define the **scalar**

$$
V_t := \|e_t\|_2^2 = \sum_{j=1}^D e_{t,j}^2.
$$

Wiggins Ch.2: a **Liapunov function** is positive definite in a neighborhood of a target set and **non-increasing** along trajectories (continuous time: $\(\dot V \le 0\)$; maps: $\(V_{n+1} \le V_n\)$ or relaxed forms).

Here $\(V_t\)$ measures **transversal energy** (distance-from-tube in normal directions). It is **not** a certified Lyapunov function for the **full** map $\(F_t\)$ without extra assumptions; it is the **template** Ch.2 uses: **one nonnegative scalar** summarizing deviation from nominal geometry.

**In code:** `V_all = (e_all ** 2).sum(dim=-1)`.

### Ch.2 + Ch.1 — Discrete contraction (map-style decrease)

On **valid** edges $\(t \to t+1\)$ (excluding skipped prompt gaps in STP mode), enforce a **soft** version of

$$
V_{t+1} \le \gamma V_t + \tau, \qquad 0 < \gamma < 1,\quad \tau \ge 0.
$$

Training minimizes violations, e.g.

$$
\ell_t = \sigma\bigl( V_{t+1} - \gamma V_t - \tau \bigr), \qquad
\sigma \in \{\mathrm{softplus}, \mathrm{ReLU}\}.
$$

This is a **discrete-time** analogue of $\(\dot V \le 0\)$ with **slack** $\(\tau\)$ (ISS-flavored bound), aligned with **maps** in Ch.1 and **Liapunov decrease** in Ch.2.

**In code:** `violation = V_curr - gamma * V_prev - tau`, masked by `valid_transitions`.

### Optional TS term (Wang et al., not Wiggins-specific)

$$
\mathcal{L}_{\mathrm{TS}} = \mathbb{E}_t\bigl[ 1 - \cos( h_{t+1}-h_t,\; h_{t+2}-h_{t+1} ) \bigr]
$$

(local curvature). Implemented in `temporal_straightening_curvature_loss`. It is **orthogonal** to the global secant + Liapunov story: **local heading** vs **transversal energy + contraction**.

### Summary table

| Wiggins (concept) | Mathematical object | Implementation |
|------------------|----------------------|----------------|
| Ch.1 Maps | Discrete index $\(t\), \(h_t\)$ | `(B,T,D)` |
| Ch.3 Invariant / structured set | $\(e_t \perp \hat v\), tube around \(\hat v\)$ | `e_all`, `d_all`, `v_norm` |
| Ch.2 Liapunov | $\(V_t = \|e_t\|^2\)$ | `V_all` |
| Ch.2 + Ch.1 | $\(V_{t+1} \le \gamma V_t + \tau\)$ | `violation`, `gamma`, `tau`, mask |

### Contrast with cosine STP / TS only

| Baseline | What it penalizes | What Lyapunov tube adds |
|----------|-------------------|-------------------------|
| Cosine STP | Angle between **two chord vectors** | Global **$\(V=\|e\|^2\)$** + **stepwise** $\(V_{t+1} \le \gamma V_t + \tau\)$ on valid edges |
| TS cosine | $\(1-\cos\)$ of **consecutive short** velocities | Same TS term optional; **tube** adds **secant + transversal + contraction** |

## Unified objective

**World model**

$$
\mathcal{L} = \mathcal{L}_{\mathrm{pred}} + \lambda_{\mathrm{tube}} \mathcal{L}_{\mathrm{Lyap}} + \lambda_{\mathrm{TS}} \mathcal{L}_{\mathrm{curv}}
$$

- Curvature loss:

$$
\mathcal{L}_{\mathrm{curv}} =
\mathrm{mean}\left(
1 - \cos(h_{t+1}-h_t,\; h_{t+2}-h_{t+1})
\right)
$$

- $\\mathcal{L}_{\mathrm{Lyap}}\$: `LyapunovTubeLoss` with `mask_valid`.

**LLM (single forward, Appendix G style)**

$$
\mathcal{L} = \mathcal{L}_{\mathrm{LM}} + \lambda_{\mathrm{cos}}\mathcal{L}_{\mathrm{STP}}^{\mathrm{(paper)}} + \lambda_{\mathrm{tube}}\mathcal{L}_{\mathrm{Lyap}}
$$

Ablate $\(\lambda_{\mathrm{cos}} \in \{0, \cdot\}\)$ to show Lyapunov tube **replaces or complements** cosine STP.

## Experiments (minimum credible set)

1. **LLM:** NL-RX-SYNTH (or paper datasets) — accuracy vs data fraction; compare baseline LM, cosine STP only, tube only, tube+cosine.
2. **WM:** goal-reaching / open-loop error — compare TS curvature only, tube only, both (use `UnifiedDynamicsRegularizer`).
3. **Diagnostics:** distribution of $\(V_t\)$ along depth and along $\(t\)$; fraction of violated Lyapunov steps; PCK of $\(\|e_t\|\)$ before/after training.
4. **Stress:** long context, large gap between Q/A (STP appendix regime).

## Code

- `dynamics_tube_loss.py`: `LyapunovTubeLoss`, `temporal_straightening_curvature_loss`, `UnifiedDynamicsRegularizer`.
- **`stp.py` integration:** pass `--dynamics_tube` (sets `linear='dynamics'`). Uses one full forward, `user_start_end` / `assistant_start_end` from the dataset, and `LyapunovTubeLoss`. Optional `--lbd_ts` adds masked local TS curvature. Hyperparameters: `--tube_gamma`, `--tube_tau`, `--lbd` (tube weight vs LM).
- **Shell:** `run_stp.py` (bash) defines `run_dynamics` and `train_three_modes_one_dataset` for regular + JEPA + dynamics on one dataset (default `NPROC=2`, `DATA_PREFIX=datasets/`).

## Related work positioning

- **Tube MPC / robust MPC:** language of “tube” around nominal trajectory (cite Mayne et al.).
- **Lyapunov networks / learning Lyapunov functions:** cite for ML + stability literature.
- **STP / TS:** position as **special cases** (cosine = alignment; TS = local curvature) subsumed in a **two-level** story: global tube + local curvature.

## Inference-time: can the LLM follow the “geometric trajectory”?

**Short answer:** Training with a Wiggins-flavored **Lyapunov/tube** loss **does not**, by itself, force autoregressive **inference** to stay on the same geometric tube. Teacher forcing defines a **different** discrete map than free-running generation: the **input** to \(F_t\) is ground-truth prefix at train time and **model samples** at test time (**distribution shift**). So Ch.1–3 **machinery** applies to **whatever** dynamical system you actually run; at inference that system is **not** identical to the training-time chain.

**What would be needed (control / feedback viewpoint, consistent with Wiggins’ emphasis on the actual map):**

1. **Closed-loop feedback at inference**  
   Treat $\(V_t\)$ (or \(\|$e_t$\|\)) as a **running output** of the generator. **Steer** generation so $\(V_t\)$ stays small, e.g.:
   - **Latent guidance:** small gradient step on logits or on last-layer hidden state each step to reduce $\(V_t\)$ (needs a defined \($v_{\mathrm{geo}}\$) or reference trajectory online—hard without a fixed prompt structure).
   - **Classifier-free / energy guidance:** add a term proportional to $\(-\nabla_{h}\log p(h \mid \text{on-tube})\)$ if you train a scorer (expensive).

2. **Reference trajectory \(h^\star_t\)**  
   If you have a **nominal** hidden trajectory (teacher model, EMA, or encoded plan), define **tracking error** $\(\tilde e_t = h_t - h^\star_t\)$ and minimize $\(\|\tilde e_t\|\)$ or a Lyapunov function of $\(\tilde e_t\)$ **during** decoding—this is **output regulation**, not just train-time regularization.

3. **Constrained decoding**  
   Reject or resample continuations that violate $\(V_{t+1} \le \gamma V_t + \tau\)$ under your metric (needs fast $\(V_t\)$ and may hurt fluency).

4. **What training *does* buy you**  
   It **shapes** the learned $\(F_t\)$ so that **on-distribution** prefixes (often close to training) **tend** to produce smaller $\(V_t\)$ and fewer violations—**implicit** geometric habits, not a **theorem** for arbitrary prompts.

**Paper-honest wording:** Position inference-time methods as **optional** “geometric control at decode time”; separate **train-time inductive bias** (Ch.2 template) from **test-time feedback** (engineering control loop).

## Ethics / honesty

State clearly: losses are **inductive biases**, not certificates of test-time stability unless you add restricted dynamics and proofs.
