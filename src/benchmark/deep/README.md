# Second attempt at deep learning: feature-supervised typing

Standalone exploration. **Does not touch the manuscript, the SI, or any canonical pipeline
output.** Everything reads from the repo root and writes only inside `second_attempt_deep/`.

## Why the first attempt died (context)

The Stage-2 FFT-Transformer (PROGRESS_REPORT §70-71) was trained end-to-end on synthetic
SN/TC/Null trajectories and reached 99% synthetic accuracy, but the matched-inflection
experiment proved it had learned exactly one feature: the inflection position t50. When the
SN/TC training classes were matched on t50 (so inflection carries zero class information),
the FFT collapsed to 50.0%, exact chance. Meanwhile (§72a) the hand-crafted theory features
scored 93.5% on the very same matched set: a genuine dynamical signal (critical slowing
down, noise structure) exists, the network just never looked for it because the t50
shortcut was cheaper.

## The new idea

Instead of handing the network a time series and hoping it discovers the dynamics, we hand
it the 46-D theory-feature space of the SI (S2) as the thing to learn. Two designs, compared:

1. **FeatMLP (floor)**: the 46-D feature vector is the *input*; a small MLP classifies
   SN/TC/Null. Supervised twin of Stage-1; should inherit the ~93% matched signal.
2. **Feature-supervised encoder (the real attempt)**: the raw 500-point trajectory is the
   input; the network is trained to *predict the 46 features* (regression head) jointly
   with the class (classification head). The features act as teachers that force the
   encoder's representation toward the dynamics.
3. **Ablation**: the identical encoder with the feature head switched off (lambda = 0).
   If (2) beats (3) on the matched set, feature supervision is what rescues the learning.

## Training data regime

Wide-parameter, no-inflection-rejection SN/TC generation (from
`matched_inflection_experiment.gen_pool`) plus a logistic Null with wide centre t0 and
stationary AR(1) noise of *varied* amplitude (so noise level is not a Null shortcut; the
Null still has no CSD by construction, per the SI benchmark). All three classes are then
**t50-matched** (equal counts per t50 bin), in the training set as well as the test set, so
the inflection shortcut carries zero class information end to end.

## Success criterion (the arbiter)

Accuracy on the t50-matched test set. Reference points:

- t50-only classifier: ~chance (by construction)
- old FFT (§71): 50.0% on 2-class matched, i.e. nothing
- 46-D nearest-centroid / logistic regression: the §72a signal, ~90%+
- the encoder counts as having *learned the dynamics* only if it lands near the
  feature-based ceiling on the matched set, far above its lambda = 0 ablation.

A natural (unmatched, wide-parameter) test set is scored as a control.

## Files

| file | role |
|---|---|
| `data.py` | generate wide SN/TC/Null pools, t50-match, split, save to `data/` |
| `features.py` | canonical 46-D features (delegates to repo modules; asserts vs `paper_figures`) |
| `models.py` | FeatMLP + TSEncoder (conv + BiLSTM, class head + 46-D regression head) |
| `train.py` | trains FeatMLP, Encoder(lambda=1), Encoder(lambda=0) on the matched train split |
| `evaluate.py` | baselines + test accuracies + feature-prediction R2 + verdict figure |
| `run_all.py` | runs the four steps in order |

Outputs land in `second_attempt_deep/runs/`.

## Run

```bash
cd /Users/Omidkh7/Downloads/Tech_Adoption_DL
python second_attempt_deep/run_all.py          # or the four scripts in order
```
