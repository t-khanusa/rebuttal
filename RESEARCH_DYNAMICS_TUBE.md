# Research blueprint: Dynamics-grounded tube regularization (beyond cosine STP / TS)

## One-sentence pitch

Replace purely geometric cosine heuristics with a **discrete-time Lyapunov-flavored transversal energy** plus optional **paper-faithful local curvature**, giving a testable stability story (Wiggins Ch.1–3) and ablations that isolate each mechanism.

## Problem with cosine-only baselines

- **STP:** \(1-\cos\) between chord vectors is a **directional** alignment proxy; it does not encode **monotone contraction** of “noise” along time or a **set** (tube) in state space.
- **TS:** \(1-\cos(v_t,v_{t+1})\) is **local**; it does not constrain **global** deviation from a long-horizon secant or Q/A-skipped semantics.

## Formal object (discrete time)

- Hidden states \(h_t \in \mathbb{R}^D\) along token or frame index \(t\).
- **Reference direction** \(v_{\mathrm{geo}}\): secant between semantic anchors (STP with gap) or first–last valid frame (WM).
- **Stitched displacements** \(d_t\) (STP mode) or \(d_t = h_t - h_{s}\) (WM mode) as in `LyapunovTubeLoss`.
- **Transversal component** \(e_t = d_t - \Pi_{v} d_t\), **Lyapunov candidate** \(V_t = \|e_t\|^2\).
- **Training surrogate:** penalize violations of \(V_{t+1} \le \gamma V_t + \tau\) on **valid** transitions (softplus/ReLU).

**Claim style (theorem sketch, not automatic truth):** If the learned map along \(t\) were a contraction in transversal coordinates with slack \(\tau\), then tube width would be bounded (discrete-time ISS flavor). Training **encourages** this; it does **not** prove stability of inference unless you add explicit model class assumptions.

## Unified objective

**World model**

\[
\mathcal{L} = \mathcal{L}_{\mathrm{pred}} + \lambda_{\mathrm{tube}} \mathcal{L}_{\mathrm{Lyap}} + \lambda_{\mathrm{TS}} \mathcal{L}_{\mathrm{curv}}
\]

- \(\mathcal{L}_{\mathrm{curv}}\): mean \(1-\cos(h_{t+1}-h_t, h_{t+2}-h_{t+1})\) on valid triples (matches TS paper).
- \(\mathcal{L}_{\mathrm{Lyap}}\): `LyapunovTubeLoss` with `mask_valid`.

**LLM (single forward, Appendix G style)**

\[
\mathcal{L} = \mathcal{L}_{\mathrm{LM}} + \lambda_{\mathrm{cos}}\mathcal{L}_{\mathrm{STP}}^{\mathrm{(paper)}} + \lambda_{\mathrm{tube}}\mathcal{L}_{\mathrm{Lyap}}
\]

Ablate \(\lambda_{\mathrm{cos}} \in \{0, \cdot\}\) to show Lyapunov tube **replaces or complements** cosine STP.

## Experiments (minimum credible set)

1. **LLM:** NL-RX-SYNTH (or paper datasets) — accuracy vs data fraction; compare baseline LM, cosine STP only, tube only, tube+cosine.
2. **WM:** goal-reaching / open-loop error — compare TS curvature only, tube only, both (use `UnifiedDynamicsRegularizer`).
3. **Diagnostics:** distribution of \(V_t\) along depth and along \(t\); fraction of violated Lyapunov steps; PCK of \(\|e_t\|\) before/after training.
4. **Stress:** long context, large gap between Q/A (STP appendix regime).

## Code

- `dynamics_tube_loss.py`: `LyapunovTubeLoss`, `temporal_straightening_curvature_loss`, `UnifiedDynamicsRegularizer`.
- Integration: call from your trainer with `h = hidden_states[-1]` and bounds or mask; multiply by \(\lambda\) and add to main loss.

## Related work positioning

- **Tube MPC / robust MPC:** language of “tube” around nominal trajectory (cite Mayne et al.).
- **Lyapunov networks / learning Lyapunov functions:** cite for ML + stability literature.
- **STP / TS:** position as **special cases** (cosine = alignment; TS = local curvature) subsumed in a **two-level** story: global tube + local curvature.

## Ethics / honesty

State clearly: losses are **inductive biases**, not certificates of test-time stability unless you add restricted dynamics and proofs.
