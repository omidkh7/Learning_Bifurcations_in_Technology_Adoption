#!/usr/bin/env python3
"""
ANALYSIS ONLY (Mahdi review 1.2 / 5.2). Quantifies how much of the real-vs-synthetic
confidence gap in figS_dl_confidence is an aggregation artifact of scoring real curves
with a seed-ensemble mean while scoring synthetic with a single seed.

Writes nothing the manuscript uses; prints tables. Run from repo root.
Proper settings (matches the current SI figure): N_PER=10000, N_SEEDS=10.
"""
import os, sys, warnings
for _v in ("OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v,"1")
warnings.filterwarnings("ignore")
import numpy as np, torch
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0,HERE); sys.path.insert(0,ROOT)
from benchmark_data import load_benchmark
from paper_figures import build_features, load_four_group
from models import FeatMLP
import train as T

N_PER, N_SEEDS = 10000, 10

def st(mx):
    return dict(median=float(np.median(mx)), gt9=float((mx>0.9).mean()),
                gt8=float((mx>0.8).mean()), lt6=float((mx<0.6).mean()))

def main():
    print(f"loading benchmark (N_PER={N_PER}) + features ...", flush=True)
    Xb,y=load_benchmark("logistic",N_PER,seed=0)
    Fb=np.nan_to_num(build_features(Xb)).astype(np.float32); y=y.astype(np.int64)
    grp,Fr,_=load_four_group(); Fr=np.nan_to_num(Fr).astype(np.float32)
    print(f"  synthetic {Fb.shape}, real {Fr.shape}", flush=True)

    Pr_all=[]           # real softmax per seed (all real curves scored by every model)
    Ps_test=[]          # (idx_test, synth softmax on that seed's own held-out test)
    for s in range(N_SEEDS):
        itr,iva,ite=T.make_splits(y,seed=1400+s)
        mu,sd=Fb[itr].mean(0),Fb[itr].std(0)+1e-9
        Fz=np.clip((Fb-mu)/sd,-8,8).astype(np.float32)
        Frz=np.clip((Fr-mu)/sd,-8,8).astype(np.float32)
        torch.manual_seed(s); np.random.seed(s)
        m,_,_=T.train_model(FeatMLP(),f"symchk/s{s}","mlp",0.0,
                            (torch.from_numpy(Fz),torch.from_numpy(y)),itr,iva)
        m.eval()
        with torch.no_grad():
            Pr_all.append(torch.softmax(m(torch.from_numpy(Frz).to(T.DEVICE)),1).cpu().numpy())
            Ps_test.append((ite, torch.softmax(m(torch.from_numpy(Fz[ite]).to(T.DEVICE)),1).cpu().numpy()))
        print(f"  seed {s} done", flush=True)

    Pr_all=np.array(Pr_all)                                    # (S, n_real, 3)

    # ---- (A) CURRENT SI PROTOCOL (the bug): real = ensemble mean; synth = seed-0 single model ----
    real_curr = st(Pr_all.mean(0).max(1))
    synth_curr = st(Ps_test[0][1].max(1))

    # ---- (B) SYMMETRIC per-seed (both sides single-model; Mahdi 5.2) ----
    real_ps  = [st(Pr_all[s].max(1)) for s in range(N_SEEDS)]
    synth_ps = [st(Ps_test[s][1].max(1)) for s in range(N_SEEDS)]
    def avg(dl,k): return float(np.mean([d[k] for d in dl]))
    def sdv(dl,k): return float(np.std([d[k] for d in dl]))

    # ---- (C) label seed-stability on real curves (context for the artifact) ----
    real_lab = Pr_all.argmax(2)                               # (S, n_real)
    from scipy import stats as sstats
    modal = sstats.mode(real_lab,axis=0,keepdims=False).count
    seedstable = float((modal/N_SEEDS).mean())

    P=lambda x:f"{100*x:4.0f}%"
    print("\n"+"="*76)
    print("CONFIDENCE PROTOCOL COMPARISON  (real four-family n=%d)"%Fr.shape[0])
    print("="*76)
    print(f"{'protocol':46s} {'median':>7s} {'>0.9':>6s} {'>0.8':>6s} {'<0.6':>6s}")
    print("-"*76)
    print(f"{'(A) CURRENT SI: real=ensemble, synth=1 seed':46s} "
          f"{real_curr['median']:7.2f} {P(real_curr['gt9'])} {P(real_curr['gt8'])} {P(real_curr['lt6'])}   <- real (in SI now)")
    print(f"{'    synth (seed 0 only)':46s} "
          f"{synth_curr['median']:7.2f} {P(synth_curr['gt9'])} {P(synth_curr['gt8'])} {P(synth_curr['lt6'])}   <- synth (in SI now)")
    print("-"*76)
    print(f"{'(B) SYMMETRIC per-seed, REAL (mean+/-sd)':46s} "
          f"{avg(real_ps,'median'):.2f}+/-{sdv(real_ps,'median'):.2f} {P(avg(real_ps,'gt9'))} {P(avg(real_ps,'gt8'))} {P(avg(real_ps,'lt6'))}")
    print(f"{'    SYMMETRIC per-seed, SYNTH (mean+/-sd)':46s} "
          f"{avg(synth_ps,'median'):.2f}+/-{sdv(synth_ps,'median'):.2f} {P(avg(synth_ps,'gt9'))} {P(avg(synth_ps,'gt8'))} {P(avg(synth_ps,'lt6'))}")
    print("-"*76)
    print("per-seed real medians:", [round(d['median'],3) for d in real_ps])
    print("per-seed synth medians:", [round(d['median'],3) for d in synth_ps])
    print(f"real-curve label seed-stability: {100*seedstable:.0f}%  (Mahdi check4c ~58%)")
    print("="*76)

if __name__=="__main__":
    main()
