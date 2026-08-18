#!/usr/bin/env python3
"""
train.py — trains the three models on the t50-MATCHED training split.

  featmlp    : FeatMLP on the 46-D features (floor)
  encoder_fs : TSEncoder, loss = CE + LAMBDA * MSE(46 standardised features)
  encoder_0  : TSEncoder, feature head off (lambda = 0) — the ablation

Splits are stratified per class 70/15/15 with a fixed seed; the SAME split indices are
saved and reused by evaluate.py. Feature standardisation (for both the MLP input and the
regression targets) is fit on the training split only.

Writes to results/benchmark/: {name}.pt checkpoints, scaler.npz, splits.npz,
history.json.
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
from models import FeatMLP, TSEncoder, multitask_loss

DATA = os.path.join(ROOT, "data", "curated", "deep")
RUNS = os.path.join(ROOT, "results", "benchmark", "deep")
os.makedirs(RUNS, exist_ok=True)

SEED = 42
LAMBDA = 0.3
BATCH = 256
MAX_EPOCHS = 80
PATIENCE = 14
LR = 1e-3
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def make_splits(y, seed=SEED, frac=(0.7, 0.15)):
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_tr = int(len(idx) * frac[0]); n_va = int(len(idx) * frac[1])
        tr.append(idx[:n_tr]); va.append(idx[n_tr:n_tr + n_va]); te.append(idx[n_tr + n_va:])
    return (np.concatenate(tr), np.concatenate(va), np.concatenate(te))


def loaders_for(tensors, idx_tr, idx_va, shuffle_train=True):
    ds = TensorDataset(*tensors)
    tr = DataLoader(torch.utils.data.Subset(ds, idx_tr.tolist()), batch_size=BATCH,
                    shuffle=shuffle_train)
    va = DataLoader(torch.utils.data.Subset(ds, idx_va.tolist()), batch_size=512)
    return tr, va


def run_epoch(model, loader, kind, lam=0.0, opt=None):
    """kind: 'mlp' (f, y) or 'enc' (x, f, y). Returns (mean CE, accuracy, mean MSE)."""
    train = opt is not None
    model.train(train)
    tot_ce = tot_mse = n = correct = 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = [b.to(DEVICE) for b in batch]
            if kind == "mlp":
                f, y = batch
                logits = model(f)
                loss = torch.nn.functional.cross_entropy(logits, y)
                ce, mse = loss.detach(), torch.tensor(0.0)
            else:
                x, f, y = batch
                logits, fp = model(x)
                loss, ce, mse = multitask_loss(logits, fp, y, f, lam)
            if train:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            bs = len(y)
            tot_ce += float(ce) * bs; tot_mse += float(mse) * bs; n += bs
            correct += int((logits.argmax(1) == y).sum())
    return tot_ce / n, correct / n, tot_mse / n


def train_model(model, name, kind, lam, tensors, idx_tr, idx_va):
    print(f"\n=== {name} (device {DEVICE}) ===")
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    tr, va = loaders_for(tensors, idx_tr, idx_va)
    hist, best_acc, best_state, bad = [], 0.0, None, 0
    for ep in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        tce, tacc, tmse = run_epoch(model, tr, kind, lam, opt)
        vce, vacc, vmse = run_epoch(model, va, kind, lam)
        sched.step()
        hist.append(dict(epoch=ep, train_ce=tce, train_acc=tacc, train_mse=tmse,
                         val_ce=vce, val_acc=vacc, val_mse=vmse))
        star = ""
        if vacc > best_acc:
            best_acc, bad = vacc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            star = " *"
        else:
            bad += 1
        print(f"  ep{ep:3d}  train acc {tacc:.3f} ce {tce:.3f} mse {tmse:.3f} | "
              f"val acc {vacc:.3f} ce {vce:.3f} mse {vmse:.3f}  ({time.time()-t0:.0f}s){star}")
        if bad >= PATIENCE:
            print(f"  early stop at epoch {ep} (best val acc {best_acc:.3f})")
            break
    model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(f"{RUNS}/{name}.pt"), exist_ok=True)   # name may contain a subdir
    torch.save(best_state, f"{RUNS}/{name}.pt")
    return model, hist, best_acc


def main():
    X = np.load(f"{DATA}/X_matched.npy").astype(np.float32)
    F = np.load(f"{DATA}/F_matched.npy").astype(np.float32)
    y = np.load(f"{DATA}/y_matched.npy").astype(np.int64)
    idx_tr, idx_va, idx_te = make_splits(y)
    np.savez(f"{RUNS}/splits.npz", train=idx_tr, val=idx_va, test=idx_te)

    mu = F[idx_tr].mean(0); sd = F[idx_tr].std(0) + 1e-9
    Fz = np.clip((F - mu) / sd, -8, 8)          # clip extreme tails for stable regression
    np.savez(f"{RUNS}/scaler.npz", mu=mu, sd=sd)

    tX = torch.from_numpy(X); tF = torch.from_numpy(Fz); ty = torch.from_numpy(y)
    histories = {}

    m, h, acc = train_model(FeatMLP(), "featmlp", "mlp", 0.0, (tF, ty), idx_tr, idx_va)
    histories["featmlp"] = h

    m, h, acc = train_model(TSEncoder(), "encoder_fs", "enc", LAMBDA, (tX, tF, ty),
                            idx_tr, idx_va)
    histories["encoder_fs"] = h

    m, h, acc = train_model(TSEncoder(), "encoder_0", "enc", 0.0, (tX, tF, ty),
                            idx_tr, idx_va)
    histories["encoder_0"] = h

    with open(f"{RUNS}/history.json", "w") as f:
        json.dump(dict(lambda_fs=LAMBDA, seed=SEED, histories=histories), f, indent=2)
    print(f"\nSaved checkpoints, splits, scaler, history to {RUNS}")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    main()
