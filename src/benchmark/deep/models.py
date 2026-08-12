#!/usr/bin/env python3
"""
models.py — the two architectures of the second attempt.

FeatMLP        : 46-D features in -> class. The floor: a supervised twin of Stage-1.
TSEncoder      : raw 500-pt trajectory in -> shared embedding -> (a) class head and
                 (b) 46-D feature-regression head. Trained with
                 loss = CE(class) + lambda * MSE(standardised features);
                 lambda = 0 is the ablation (plain end-to-end, expected to fail the
                 matched-inflection test like the old FFT did).

The encoder sees both the normalised curve and its first difference as a second channel:
the CSD signal lives in the residual noise, which the difference channel exposes directly.
"""
import torch
import torch.nn as nn

N_FEATURES = 46
N_CLASSES = 3


class FeatMLP(nn.Module):
    def __init__(self, n_in=N_FEATURES, n_classes=N_CLASSES, hidden=(128, 64), p_drop=0.2):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p_drop)]
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, f):
        return self.net(f)


class TSEncoder(nn.Module):
    """Conv stack + BiLSTM over (x, dx) channels; class head + feature head."""

    def __init__(self, n_classes=N_CLASSES, n_features=N_FEATURES, emb=128, p_drop=0.15):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=9, padding=4), nn.ReLU(),
            nn.MaxPool1d(2),                                    # 500 -> 250
            nn.Conv1d(32, 64, kernel_size=7, padding=3), nn.ReLU(),
            nn.MaxPool1d(2),                                    # 250 -> 125
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Dropout(p_drop),
        )
        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        self.proj = nn.Sequential(nn.Linear(2 * 64 * 2, emb), nn.ReLU(), nn.Dropout(p_drop))
        self.head_class = nn.Linear(emb, n_classes)
        self.head_feat = nn.Linear(emb, n_features)

    def forward(self, x):
        # x: (B, 500) normalised curve
        dx = torch.diff(x, dim=1, prepend=x[:, :1])
        h = self.conv(torch.stack([x, dx], dim=1))              # (B, 64, 125)
        out, _ = self.lstm(h.transpose(1, 2))                   # (B, 125, 128)
        z = torch.cat([out.mean(dim=1), out[:, -1]], dim=1)     # mean + last: (B, 256)
        z = self.proj(z)
        return self.head_class(z), self.head_feat(z)


def multitask_loss(logits, feat_pred, y, feat_true, lam):
    ce = nn.functional.cross_entropy(logits, y)
    if lam <= 0:
        return ce, ce.detach(), torch.tensor(0.0, device=logits.device)
    mse = nn.functional.mse_loss(feat_pred, feat_true)
    return ce + lam * mse, ce.detach(), mse.detach()
