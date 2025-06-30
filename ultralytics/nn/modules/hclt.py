"""
HCLT-YOLO custom modules
–––––––––––––––––––––––––
• CRepvit   – CNN-Transformer hybrid block that fuses RepViT token-/channel-mixing
              with a MobileNetV2 style stem (§III-B in the paper[1]).
• DG_C2f    – Lightweight C2f replacement that combines GhostConv and depth-wise
              separable convs to reduce FLOPs (§III-C in the paper[1]).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv, DWConv, RepConv, GhostConv


# --------------------------------------------------------------------------- #
# 1. RepViT-style token mixer (depth-wise 3×3 + SE)                           #
# --------------------------------------------------------------------------- #
class _RepViTTokenMixer(nn.Module):
    """Single RepViT token-mixer block (Fig. 3 in the paper[1])."""

    def __init__(self, c: int):
        super().__init__()
        self.dw = DWConv(c, c, k=3, s=1)  # depth-wise 3×3
        # Squeeze-and-Excitation (ratio 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // 4, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(c // 4, c, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)
        return x * self.se(y)


# --------------------------------------------------------------------------- #
# 2.  C-Repvit backbone block                                                 #
# --------------------------------------------------------------------------- #
class CRepvit(nn.Module):
    """
    ELAN-inspired cascade of RepViT token- and channel-mixers (§III-B[1]).
    Args
    ----
    c1 : input channels
    c2 : output channels
    n  : number of RepViT sub-blocks
    """

    def __init__(self, c1: int, c2: int, n: int = 1):
        super().__init__()
        self.stem = Conv(c1, c2, k=1, s=1)  # MobileNetV2-style point-wise conv

        blocks = []
        for _ in range(n):
            blocks += [
                _RepViTTokenMixer(c2),  # token mixer
                RepConv(c2, c2, k=3, s=1),  # channel mixer (re-parameterisable)
            ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.stem(x))


# --------------------------------------------------------------------------- #
# 3.  DG-C2f neck block (Ghost + DW + C2f fusion)                             #
# --------------------------------------------------------------------------- #
class DG_C2f(nn.Module):
    """
    Ghost-based lightweight alternative to C2f (§III-C[1]).
    Keeps *exactly* the same IO signature as the standard C2f block.
    """

    def __init__(self, c1: int, c2: int, n: int = 1):
        """
        c1 : input channels
        c2 : output channels
        n  : number of internal GhostConv repetitions
        """
        super().__init__()
        self.cv1 = GhostConv(c1, c2, k=3, s=1)
        self.m = nn.ModuleList([GhostConv(c2, c2, k=3, s=1) for _ in range(n)])
        self.cv2 = Conv(c2 * (n + 1), c2, k=1, s=1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        for layer in self.m:
            y.append(layer(y[-1]))
        return self.cv2(torch.cat(y, 1))
