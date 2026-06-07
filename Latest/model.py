
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Squeeze-and-Excitation gate
# =============================================================================

class SEGate(nn.Module):

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        s = x.mean(dim=2)                   # global avg pool → (B, C)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))      # (B, C)
        return x * s.unsqueeze(2)           # broadcast → (B, C, T)


# =============================================================================
# Residual block with SE attention
# =============================================================================

class ResBlock(nn.Module):
  

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 7,
        se_reduction: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        pad = kernel_size // 2

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)

        # Shortcut: 1×1 conv when channels differ
        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm1d(out_ch),
            )
            if in_ch != out_ch
            else nn.Identity()
        )

        self.se      = SEGate(out_ch, reduction=se_reduction)
        self.drop    = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.relu    = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)

        out = self.relu(out + identity)
        out = self.drop(out)
        return out


# =============================================================================
# TinyResECG-v2
# =============================================================================

class TinyResECG(nn.Module):
  

    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()

        # Stem: lightweight initial conv to create a richer feature map
        # before the residual blocks
        self.stem = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )

        # Block 1: 16 → 16, then downsample
        self.block1 = ResBlock(16, 16, kernel_size=7, se_reduction=4, dropout=0.0)
        self.pool1  = nn.MaxPool1d(2)           # T: 1280 → 640

        # Block 2: 16 → 32, then downsample
        self.block2 = ResBlock(16, 32, kernel_size=7, se_reduction=4, dropout=0.1)
        self.pool2  = nn.MaxPool1d(2)           # T: 640 → 320

        # Block 3: 32 → 40, then downsample
        self.block3 = ResBlock(32, 40, kernel_size=5, se_reduction=4, dropout=0.1)
        self.pool3  = nn.MaxPool1d(2)           # T: 320 → 160

        # Block 4: 40 → 40 (same channels, no downsample)
        self.block4 = ResBlock(40, 40, kernel_size=5, se_reduction=4, dropout=0.0)

        # Global average pooling → 40-dim vector
        self.gap     = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(40, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.block4(x)

        x = self.gap(x).squeeze(-1)     # (B, 48)
        x = self.dropout(x)
        x = self.fc(x)                  # (B, 2)
        return x


# =============================================================================
# Quick sanity check
# =============================================================================

if __name__ == "__main__":
    model = TinyResECG()
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")
    assert total < 50_000, f"Parameter budget exceeded: {total}"

    dummy = torch.randn(4, 2, 1280)
    out   = model(dummy)
    print(f"Output shape: {out.shape}")   # (4, 2)
    print("Model OK.")
