"""Feature Alignment Module for knowledge editing via SAE-mediated injection.

Maps the small addition model's residual stream into Qwen3-4B's MLP-output space at a
chosen injection layer, learning a bottleneck θ that aligns with the big model's SAE
feature geometry (alignment projection P, precomputed once via PCA).

Architecture
------------
Given:
  f_s   ∈ ℝ^{d_s}          — small model residual stream at '=' position
  f_B   ∈ ℝ^{d_tc}         — big model SAE feature activations at inject_layer
  W_dec ∈ ℝ^{d_tc × d_m}   — big model SAE decoder

Compute:
  decoded_f_B  = f_B @ W_dec  ∈ ℝ^{d_m}           (big model SAE output in residual space)
  z̃           = theta(f_s)   ∈ ℝ^{d_mid}          (bottleneck)
  φ_l          = P @ decoded_f_B ∈ ℝ^{d_mid}       (PCA-projected big SAE output)
  L_align      = 1 − cos(z̃, φ_l)                   (alignment loss)
  inject       = w_out(z̃)    ∈ ℝ^{d_m}             (injection vector)

Injection modes:
  Replace:  hook_mlp_out ← inject
  Add:      hook_mlp_out ← hook_mlp_out + inject
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class FeatureAlignmentModule(nn.Module):
    """Learnable bottleneck that maps a small model's features into a large model's space.

    Parameters
    ----------
    d_s:
        Dimension of the small model's representation (residual stream size).
    d_mid:
        Bottleneck dimension.  Alignment projection P lives in ℝ^{d_mid × d_m}.
    d_model_large:
        Residual stream dimension of the large (Qwen3-4B) model.
    align_proj:
        Fixed PCA projection P ∈ ℝ^{d_mid × d_model_large}, precomputed via
        :py:meth:`compute_align_proj`.  Pass ``None`` to skip the alignment loss
        (useful for a quick sanity-check forward pass before setup is done).
    """

    def __init__(
        self,
        d_s: int = 256,
        d_mid: int = 256,
        d_model_large: int = 2560,
        align_proj: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.d_s = d_s
        self.d_mid = d_mid
        self.d_model_large = d_model_large

        # θ: small-model features → bottleneck
        self.theta = nn.Sequential(
            nn.Linear(d_s, max(d_s, d_mid * 2)),
            nn.GELU(),
            nn.Linear(max(d_s, d_mid * 2), d_mid),
        )

        # w_out: bottleneck → large-model residual stream
        self.w_out = nn.Linear(d_mid, d_model_large, bias=False)

        # Fixed alignment projection (set after precomputation, never trained)
        if align_proj is not None:
            self.register_buffer("align_proj", align_proj)
        else:
            self.register_buffer("align_proj", None)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, f_s: torch.Tensor) -> torch.Tensor:
        """Map small model features to injection vector.

        Parameters
        ----------
        f_s:
            Shape ``(batch, d_s)`` or ``(d_s,)``.

        Returns
        -------
        torch.Tensor
            Shape matching ``f_s`` leading dims, last dim = ``d_model_large``.
        """
        f_s = F.normalize(f_s, dim=-1)  # unit-norm: remove magnitude scale
        z = self.theta(f_s)  # (..., d_mid)
        return self.w_out(z)  # (..., d_model_large)

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    def bottleneck(self, f_s: torch.Tensor) -> torch.Tensor:
        """Return bottleneck representation z̃ = theta(f_s)."""
        return self.theta(f_s)

    def align_loss(
        self,
        f_s: torch.Tensor,
        decoded_f_B: torch.Tensor,
    ) -> torch.Tensor:
        """Cosine alignment loss between bottleneck and PCA-projected SAE output.

        L_align = 1 − mean(cos(theta(f_s), P @ decoded_f_B))

        Parameters
        ----------
        f_s:
            Shape ``(batch, d_s)``.
        decoded_f_B:
            Shape ``(batch, d_model_large)`` — big model SAE output decoded to
            residual stream space (``f_B_activations @ W_dec``).

        Returns
        -------
        Scalar tensor (mean over batch).
        """
        if self.align_proj is None:
            raise RuntimeError("align_proj is None — run --setup first to compute PCA projection")

        z = self.theta(F.normalize(f_s, dim=-1))  # (B, d_mid)
        phi = decoded_f_B @ self.align_proj.T  # (B, d_mid)
        cos = F.cosine_similarity(z, phi, dim=-1)  # (B,)
        return (1.0 - cos).mean()

    # ------------------------------------------------------------------
    # Precomputation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_align_proj(
        decoded_f_B_samples: torch.Tensor,
        d_mid: int,
        method: str = "pca",
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the fixed alignment projection P.

        Two methods are supported:

        ``"pca"`` *(default)*
            Unsupervised. Returns the top-``d_mid`` right singular vectors of the
            centred ``decoded_f_B_samples`` matrix.  No labels required.

        ``"probe"``
            Supervised.  Trains a one-vs-rest logistic regression probe for each
            unique carry count in ``labels`` and stacks the normalised weight
            vectors as discriminant directions.  Each direction directly answers
            "which way in decoded_f_B space predicts carry count k?".  Any
            remaining budget (when the number of classes is smaller than
            ``d_mid``) is filled with PCA directions orthogonal to the probe
            subspace.  Requires ``labels``.

        Parameters
        ----------
        decoded_f_B_samples:
            Shape ``(N, d_model_large)`` — decoded SAE outputs collected over the
            training set.
        d_mid:
            Number of projection directions (output rows of P).
        method:
            ``"pca"`` or ``"probe"``.
        labels:
            Integer tensor of shape ``(N,)`` — carry count per sample.
            Required for ``"probe"``.

        Returns
        -------
        P : torch.Tensor of shape ``(d_mid, d_model_large)``
        """
        import numpy as np

        X_np = decoded_f_B_samples.float().cpu().numpy()
        N, d_m = X_np.shape

        log.info(
            "Computing align proj [%s] from %d samples, d_model_large=%d → d_mid=%d",
            method,
            N,
            d_m,
            d_mid,
        )

        def _pca_directions(Z: np.ndarray, k: int) -> np.ndarray:
            """Top-k right singular vectors of centred Z, shape (k, d_m)."""
            Z = Z - Z.mean(axis=0, keepdims=True)
            _, _, Vt = np.linalg.svd(Z, full_matrices=False)
            return Vt[:k]

        if method == "pca":
            P_np = _pca_directions(X_np, d_mid)

        elif method == "probe":
            if labels is None:
                raise ValueError("method='probe' requires labels (carry counts per sample)")

            from sklearn.linear_model import LogisticRegression

            y = labels.cpu().numpy().astype(int)
            classes = np.unique(y)
            probe_dirs = []
            for cls in classes:
                binary_y = (y == cls).astype(int)
                if binary_y.sum() < 2 or (1 - binary_y).sum() < 2:
                    continue
                lr = LogisticRegression(max_iter=500, C=1.0)
                lr.fit(X_np, binary_y)
                w = lr.coef_[0]
                probe_dirs.append(w / (np.linalg.norm(w) + 1e-8))

            probe_dirs_np = np.stack(probe_dirs, axis=0)  # (n_probes, d_m)
            n_probe = probe_dirs_np.shape[0]
            log.info("Probe produced %d direction(s) from %d classes", n_probe, len(classes))

            if n_probe >= d_mid:
                P_np = probe_dirs_np[:d_mid]
            else:
                # Pad with PCA directions orthogonal to the probe subspace
                Q, _ = np.linalg.qr(probe_dirs_np.T, mode="reduced")  # (d_m, n_probe)
                X_resid = X_np - X_np @ Q @ Q.T  # remove subspace
                pca_dirs = _pca_directions(X_resid, d_mid - n_probe)
                P_np = np.concatenate([probe_dirs_np, pca_dirs], axis=0)

        else:
            raise ValueError(f"Unknown method: {method!r}. Choose 'pca' or 'probe'.")

        P = torch.from_numpy(P_np.astype(np.float32))
        log.info("Align proj done — P shape: %s, method=%s", tuple(P.shape), method)
        return P

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "d_s": self.d_s,
                "d_mid": self.d_mid,
                "d_model_large": self.d_model_large,
                "state_dict": self.state_dict(),
            },
            path,
        )
        log.info("Saved FeatureAlignmentModule to %s", path)

    @classmethod
    def load(cls, path: str | Path, device: torch.device | str = "cpu") -> FeatureAlignmentModule:
        path = Path(path)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        state_dict = ckpt["state_dict"]
        align_proj = state_dict.get("align_proj", None)
        module = cls(
            d_s=ckpt["d_s"],
            d_mid=ckpt["d_mid"],
            d_model_large=ckpt["d_model_large"],
            align_proj=align_proj,
        )
        module.load_state_dict(state_dict)
        module.to(device)
        log.info(
            "Loaded FeatureAlignmentModule from %s (d_s=%d, d_mid=%d)",
            path,
            module.d_s,
            module.d_mid,
        )
        return module
