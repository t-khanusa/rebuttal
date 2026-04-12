"""
Dynamics-Grounded Tube Regularization for Representation Learning
===================================================================

This module formalizes objectives beyond cosine heuristics used in Semantic Tube
Prediction (STP) and Temporal Straightening (TS), using discrete-time
Lyapunov-flavored constraints on *transversal* energy relative to a nominal
direction (a 1D reference in hidden space).

Research framing (paper outline)
--------------------------------
1. **Discrete plant.** Teacher-forced LM or world model yields a sequence of
   hidden states h_0,...,h_{T-1}. Index t is discrete time.

2. **Nominal manifold (working object).** Fix a reference direction v_geo in R^D
   (secant between semantic anchors for STP-with-gap, or first-to-last valid
   frame for continuous trajectories). The *tube* is the set of states whose
   displacement from a pivot lies almost in span{v_geo} — equivalently small
   normal component e_t after orthogonal projection.

3. **Lyapunov candidate (transversal).** V_t = ||e_t||^2 / (||v_geo||^2 + eps),
   i.e. orthogonal energy relative to squared secant length. This is **scale-invariant**
   if h is rescaled (e and v_geo scale together) and avoids both O(D) sums and
   arbitrary per-dimension means. This plays
   the role of a Lyapunov function for *transverse* dynamics (Wiggins Ch.2
   spirit), not stability of full closed-loop LM dynamics.

4. **Discrete contraction (ISS-flavored).** Enforce soft constraints
   V_{t+1} <= gamma * V_t + tau on *valid* transitions (skipping prompt gaps
   for STP). Training minimizes softplus/reLU violations — a differentiable
   surrogate for a one-step decrease condition along the trajectory.

5. **Optional TS paper term.** Local curvature as in Wang et al. (Temporal
   Straightening): 1 - cos(v_t, v_{t+1}) for consecutive velocities
   v_t = h_{t+1} - h_t. This is *orthogonal* to the global secant tube and
   should be exposed as a separate coefficient in a unified objective.

Suggested combined loss for a world model::
    L = L_pred + lambda_tube * L_Lyapunov_tube + lambda_ts * L_curvature

Suggested combined loss for LLM (single forward, Appendix G style)::
    L = L_LM + lambda_stp_cos * L_cosine_STP   # optional baseline
        + lambda_tube * L_Lyapunov_tube(bounds=...)

References
----------
- Huang et al., Semantic Tube Prediction (geodesic / perpendicular noise).
- Wang et al., Temporal Straightening (local cosine of consecutive velocities).
- Wiggins, *Introduction to Applied Nonlinear Dynamical Systems and Chaos*
  (Ch.1 stability & maps, Ch.2 Lyapunov functions, Ch.3 invariant manifolds).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union


def temporal_straightening_curvature_loss(
    h: torch.Tensor,
    mask_valid: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Paper-faithful local curvature term (Wang et al.): mean of 1 - cos(v_t, v_{t+1})
    with v_t = h_{t+1} - h_t. Optionally average only over transitions where both
    edges lie inside a valid mask.

    Parameters
    ----------
    h : (B, T, D)
    mask_valid : (B, T) bool or 0/1; if set, require mask[b,t] & mask[b,t+1] & mask[b,t+2]
    """
    if h.size(1) < 3:
        return h.sum() * 0.0

    v0 = h[:, 1:-1] - h[:, :-2]
    v1 = h[:, 2:] - h[:, 1:-1]
    cos = F.cosine_similarity(v0, v1, dim=-1, eps=eps)
    loss_t = 1.0 - cos

    if mask_valid is not None:
        m = mask_valid.bool()
        # transition t uses indices t, t+1, t+2 in original h
        ok = m[:, :-2] & m[:, 1:-1] & m[:, 2:]
        denom = ok.sum().clamp(min=1)
        return (loss_t * ok.float()).sum() / denom

    return loss_t.mean()


class LyapunovTubeLoss(nn.Module):
    """
    Lyapunov-regularized tube loss: orthogonal energy (normalized by ||v_geo||^2)
    relative to a secant, plus discrete contraction V_{t+1} <= gamma V_t + tau
    on valid transitions.

    Modes
    -----
    * **STP / LLM (gap skip):** pass ``user_bounds`` and ``assistant_bounds`` as
      length-B lists of (start, end) **inclusive** token indices in ``h``.
    * **Continuous trajectory:** pass ``mask_valid`` (B, T); first/last True
      per row define secant endpoints; all transitions between them are valid.
    """

    def __init__(
        self,
        gamma: float = 0.95,
        tau: float = 1e-3,
        use_softplus: bool = True,
    ):
        super().__init__()
        self.gamma = gamma
        self.tau = tau
        self.use_softplus = use_softplus

    def forward(
        self,
        h: torch.Tensor,
        mask_valid: Optional[torch.Tensor] = None,
        user_bounds: Optional[List[Tuple[int, int]]] = None,
        assistant_bounds: Optional[List[Tuple[int, int]]] = None,
        *,
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        B, T, D = h.shape
        device = h.device

        d_all = torch.zeros_like(h)
        v_geodesic = torch.zeros(B, D, device=device, dtype=h.dtype)
        valid_transitions = torch.zeros(B, T - 1, device=device, dtype=torch.bool)

        if user_bounds is not None and assistant_bounds is not None:
            if len(user_bounds) != B or len(assistant_bounds) != B:
                raise ValueError("bounds must have length B")
            for i in range(B):
                u_s, u_e = user_bounds[i]
                a_s, a_e = assistant_bounds[i]
                v_user = h[i, u_e] - h[i, u_s]
                d_all[i, u_s : u_e + 1] = h[i, u_s : u_e + 1] - h[i, u_s]
                d_all[i, a_s : a_e + 1] = v_user + (h[i, a_s : a_e + 1] - h[i, a_s])
                v_geodesic[i] = v_user + (h[i, a_e] - h[i, a_s])
                valid_transitions[i, u_s:u_e] = True
                valid_transitions[i, a_s:a_e] = True

        elif mask_valid is not None:
            m = mask_valid.bool()
            s_indices = m.long().argmax(dim=1)
            flipped = m.flip(dims=[1])
            t_indices = T - 1 - flipped.long().argmax(dim=1)
            batch_idx = torch.arange(B, device=device)
            h_s = h[batch_idx, s_indices]
            h_t = h[batch_idx, t_indices]
            v_geodesic = h_t - h_s
            d_all = h - h_s.unsqueeze(1)
            time_steps = torch.arange(T - 1, device=device).unsqueeze(0).expand(B, -1)
            valid_transitions = (time_steps >= s_indices.unsqueeze(1)) & (
                time_steps < t_indices.unsqueeze(1)
            )
        else:
            raise ValueError(
                "Provide mask_valid (continuous / WM) or user_bounds and assistant_bounds (STP)."
            )

        v_norm = F.normalize(v_geodesic, p=2, dim=1, eps=1e-8)
        dot = (d_all * v_norm.unsqueeze(1)).sum(dim=-1, keepdim=True)
        p_all = dot * v_norm.unsqueeze(1)
        e_all = d_all - p_all
        # Dimensionless V: transversal energy vs reference chord energy (scale-invariant in h).
        v_geo_sq_norm = (v_geodesic**2).sum(dim=-1).clamp(min=1e-8)
        e_sq_norm = (e_all**2).sum(dim=-1)
        V_all = e_sq_norm / v_geo_sq_norm.unsqueeze(1)

        V_prev = V_all[:, :-1]
        V_curr = V_all[:, 1:]
        violation = V_curr - self.gamma * V_prev - self.tau
        if self.use_softplus:
            loss_matrix = F.softplus(violation)
        else:
            loss_matrix = F.relu(violation)

        masked = loss_matrix * valid_transitions.float()
        num_valid = valid_transitions.sum()
        if num_valid == 0:
            z = h.sum() * 0.0
            if return_diagnostics:
                zero = torch.zeros((), device=device, dtype=h.dtype)
                return z, {
                    "mean_V_curr": zero,
                    "mean_V_prev": zero,
                    "delta_mean_V": zero,
                    "mean_raw_violation": zero,
                    "viol_rate": zero,
                    "tube_softplus_mean": zero,
                }
            return z

        num_f = num_valid.to(h.dtype)
        vf = valid_transitions.to(h.dtype)
        mean_V_curr = (V_curr * vf).sum() / num_f
        mean_V_prev = (V_prev * vf).sum() / num_f
        # Along long spans, marginal means of V[t] vs V[t+1] over edges are often near-equal
        # (differ only by boundary terms); use delta_mean_V to see small drift.
        delta_mean_V = mean_V_curr - mean_V_prev
        mean_raw_violation = (violation * vf).sum() / num_f
        viol_pos = (violation > 0).to(h.dtype) * vf
        viol_rate = viol_pos.sum() / num_f
        tube_softplus_mean = masked.sum() / num_f
        loss = tube_softplus_mean

        if return_diagnostics:
            return loss, {
                "mean_V_curr": mean_V_curr.detach(),
                "mean_V_prev": mean_V_prev.detach(),
                "delta_mean_V": delta_mean_V.detach(),
                "mean_raw_violation": mean_raw_violation.detach(),
                "viol_rate": viol_rate.detach(),
                "tube_softplus_mean": tube_softplus_mean.detach(),
            }
        return loss


class UnifiedDynamicsRegularizer(nn.Module):
    """
    Single wrapper for: (1) Lyapunov tube loss, (2) optional TS curvature loss.

    total = w_tube * L_tube + w_ts * L_ts_curvature
    L_ts_curvature is only applied when ``mask_valid`` is given and w_ts > 0.
    """

    def __init__(
        self,
        w_tube: float = 1.0,
        w_ts_curvature: float = 0.0,
        gamma: float = 0.95,
        tau: float = 1e-3,
        use_softplus: bool = True,
    ):
        super().__init__()
        self.w_tube = w_tube
        self.w_ts_curvature = w_ts_curvature
        self.tube = LyapunovTubeLoss(
            gamma=gamma, tau=tau, use_softplus=use_softplus
        )

    def forward(
        self,
        h: torch.Tensor,
        mask_valid: Optional[torch.Tensor] = None,
        user_bounds: Optional[List[Tuple[int, int]]] = None,
        assistant_bounds: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[torch.Tensor, dict]:
        tube_loss = self.tube(
            h,
            mask_valid=mask_valid,
            user_bounds=user_bounds,
            assistant_bounds=assistant_bounds,
        )
        ts_loss = torch.zeros((), device=h.device, dtype=h.dtype)
        if self.w_ts_curvature > 0 and mask_valid is not None:
            ts_loss = temporal_straightening_curvature_loss(h, mask_valid=mask_valid)

        total = self.w_tube * tube_loss + self.w_ts_curvature * ts_loss
        info = {
            "loss_total": total.detach(),
            "loss_tube": tube_loss.detach(),
            "loss_ts_curvature": ts_loss.detach(),
        }
        return total, info


__all__ = [
    "LyapunovTubeLoss",
    "UnifiedDynamicsRegularizer",
    "temporal_straightening_curvature_loss",
]


if __name__ == "__main__":
    B, T, D = 2, 8, 16
    h = torch.randn(B, T, D, requires_grad=True)
    m = torch.zeros(B, T, dtype=torch.bool)
    m[:, 2:7] = True
    L_tube = LyapunovTubeLoss()(h, mask_valid=m)
    L_ts = temporal_straightening_curvature_loss(h, mask_valid=m)
    L_uni, info = UnifiedDynamicsRegularizer(w_ts_curvature=0.5)(h, mask_valid=m)
    L_tube.backward(retain_graph=True)
    assert L_tube.shape == ()
    assert info["loss_tube"].numel() == 1
    assert h.grad is not None and h.grad.abs().sum() > 0, "tube loss must backprop into h"

    h2 = torch.randn(1, 12, 16, requires_grad=True)
    L2, diag = LyapunovTubeLoss()(
        h2,
        user_bounds=[(1, 3)],
        assistant_bounds=[(5, 9)],
        return_diagnostics=True,
    )
    assert "delta_mean_V" in diag
    h2.grad = None
    L2.backward()
    assert h2.grad is not None and h2.grad.abs().sum() > 0, "STP bounds path must backprop into h"
