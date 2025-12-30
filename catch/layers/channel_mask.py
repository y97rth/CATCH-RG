"""
@author: EmpyreanMoon
@create: 2024-09-02 17:32

@description:
Robust channel mask generator (CATCH-RG, revised)

(+) Automatic channel reliability estimation
(+) Reliability-aware soft gating for missing or noisy channels
(+) Identity fallback when all channels are globally inactive
(+) Input/output interface is identical to the original implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class channel_mask_generator(nn.Module):
    def __init__(self, input_size, n_vars):
        super(channel_mask_generator, self).__init__()

        # (modified) Same generator structure as original,
        # but behavior is changed via zero initialization + reliability modulation
        self.generator = nn.Sequential(
            nn.Linear(input_size * 2, n_vars, bias=False),
            nn.Sigmoid()
        )

        # (+) Initialize generator to identity-like behavior at the start of training
        with torch.no_grad():
            self.generator[0].weight.zero_()

        self.n_vars = n_vars

        # (+) Channel reliability estimation head
        # Learns a soft confidence score per channel from simple statistics
        self.reliability_head = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self.eps = 1e-6

    def forward(self, x):
        """
        x: [(batch_size * patch_num), n_vars, patch_size]
        return: [Bp, C, C] binary channel interaction mask
        """

        Bp, C, P = x.shape

        # (+) Numerical safety for NaN / Inf / missing values
        x_safe = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # (+) Detect globally inactive samples (all channels nearly zero)
        down_flag = (x_safe.abs().sum(dim=(1, 2)) < 1e-12)  # [Bp]

        # (modified) Base Bernoulli probability matrix
        # Original: directly used for sampling
        # Revised: later modulated by reliability gating
        distribution_matrix = self.generator(x_safe).clamp(self.eps, 1 - self.eps)

        # (+) Channel-wise statistics for reliability estimation
        var = x_safe.var(dim=-1, unbiased=False).clamp_min(1e-8)
        logvar = torch.log(var)
        absmean = x_safe.abs().mean(dim=-1)
        diff = (x_safe[..., 1:] - x_safe[..., :-1]).abs().max(dim=-1).values
        zero_ratio = (x_safe.abs() < 1e-6).float().mean(dim=-1)

        stats = torch.stack(
            [logvar, absmean, diff, zero_ratio],
            dim=-1
        )  # [Bp, C, 4]

        # (+) Soft reliability score per channel
        r = self.reliability_head(stats).squeeze(-1).clamp_min(0.05)  # [Bp, C]

        # (+) Pairwise reliability gating via outer product
        r_outer = r.unsqueeze(-1) * r.unsqueeze(-2)                  # [Bp, C, C]

        # (+) Reliability-modulated Bernoulli parameters
        distribution_matrix = (distribution_matrix * r_outer).clamp(self.eps, 1 - self.eps)

        # (modified) Straight-through Bernoulli sampling with Gumbel noise
        resample_matrix = self._bernoulli_gumbel_rsample(distribution_matrix)

        # (unchanged) Enforce diagonal = 1 (self-channel always preserved)
        inverse_eye = (1 - torch.eye(self.n_vars, device=x.device)).to(resample_matrix.dtype)
        diag = torch.eye(self.n_vars, device=x.device).to(resample_matrix.dtype)

        resample_matrix = torch.einsum("bcd,cd->bcd", resample_matrix, inverse_eye)
        resample_matrix = resample_matrix + diag

        # (+) Identity fallback when all channels are globally down
        # This fully blocks cross-channel propagation in degenerate cases
        if down_flag.any():
            I = diag.unsqueeze(0).expand(Bp, -1, -1)
            df = down_flag.view(Bp, 1, 1)
            resample_matrix = torch.where(df, I, resample_matrix)

        return resample_matrix

    def _bernoulli_gumbel_rsample(self, p):
        """
        Straight-through Bernoulli sampler with Gumbel noise
        Input / Output shape: [B, C, D]
        """

        p = p.clamp(self.eps, 1 - self.eps)

        # (modified) Logit computation with numerical stability
        logit = torch.log(p) - torch.log1p(-p)

        # (+) Gumbel noise injection
        g = -torch.log(-torch.log(torch.rand_like(logit)))

        # (+) Soft Bernoulli sample
        y = torch.sigmoid(logit + g)

        # (+) Two-class representation for straight-through estimation
        y2 = torch.stack([y, 1.0 - y], dim=-1)

        # (+) Hard sampling (forward) with soft gradients (backward)
        hard_idx = y2.argmax(dim=-1)
        y2_hard = F.one_hot(hard_idx, num_classes=2).to(y2.dtype)
        y2_st = y2 + (y2_hard - y2).detach()

        # (unchanged) Use the "selected" probability as the binary mask
        return y2_st[..., 0]
