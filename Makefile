# Reproduction driver. Run from the repo root: `make all` (everything except torch),
# `make deep` (DL scripts, needs torch). Stage-by-stage targets below.
#
# Scripts use flat imports (they were developed in one folder); PYPATH makes them
# resolve across the src/ subdirectories without touching any import statement.

PY      := python3
PYPATH  := src/curation:src/features:src/synthetic:src/benchmark:src/ews:src/analysis:src/figures
RUN     := PYTHONPATH=$(PYPATH) $(PY)

.PHONY: all dirs tables figures si si-slow audit deep deep-data deep-proper clean-figures

all: tables figures si si-slow audit

# Output directories (some scripts create their own, some assume they exist)
dirs:
	mkdir -p results/unsup/bifurcation_explore results/unsup/shape_diagnostic \
	         results/unsup/real_world results/unsup/benchmark \
	         results/figures results/logs results/benchmark \
	         figures/main figures/si data/curated

# ---- Derived tables (inputs to Fig. 4 and the S7 growth figures; run first) ----
tables: dirs
	$(RUN) src/analysis/growth_compare.py
	$(RUN) src/analysis/compare_4groups.py

# ---- Main-text + core SI figures ----
# fig1_4panel_proposal.py reads results/benchmark/benchmark_dl_proper.json
# (committed artifact; regenerate with `make deep`).
figures: tables
	$(RUN) src/figures/paper_figures.py          # Figs 1-4 + figS_feature_space + figS_benchmark (~5-15 min)
	$(RUN) src/figures/paper_si_figures.py       # S7 growth + CDR-source figures
	$(RUN) src/figures/fig1_4panel_proposal.py   # Fig 1 four-panel

# ---- SI robustness figures (fast-to-moderate) ----
si: tables
	$(RUN) src/curation/no_transition_fraction.py    # Fig S1
	$(RUN) src/benchmark/benchmark_feature_spaces.py
	$(RUN) src/benchmark/benchmark_feature_analysis.py
	$(RUN) src/analysis/group_by_group.py
	$(RUN) src/features/feature_redundancy.py
	$(RUN) src/features/feature_ablation.py
	$(RUN) src/analysis/subset_clustering.py
	$(RUN) src/analysis/finite_time_scaling.py
	$(RUN) src/ews/ews_sensitivity.py                # ~2 min
	$(RUN) src/analysis/cdr_policy_stringency.py     # S8
	$(RUN) src/analysis/pca_bimodality_prune.py      # S4 (needs diptest)
	$(RUN) src/ews/decline_ews_si.py                 # S3 Fig S6
	$(RUN) src/ews/bic_detection.py                  # S3 Fig S5 (`plot` restyles from cache)
	$(RUN) src/analysis/si_continuum_tiers.py        # S9 Fig figS_continuum_tiers (needs diptest)

# ---- SI slow jobs ----
si-slow: tables
	$(RUN) src/synthetic/null_sensitivity.py         # ~10-30 min (mixture fits)
	$(RUN) src/synthetic/null_ladder.py              # null-construction ladder
	$(RUN) src/ews/ews_null_control.py               # ~10 min (`figure` redraws from CSV)
	$(RUN) src/analysis/mc_mu_observed.py            # ~1-2 h (`figure` redraws from CSV)

# ---- Audit record (old-vs-new null comparison table) ----
audit: tables
	$(RUN) src/benchmark/audit_benchmark_rerun.py

# ---- Deep-learning SI additions (requires torch) ----
# Pools in data/curated/deep/ are shipped; `deep` trains on them, writes the JSON + results
# figures, and copies the two DL SI figures into figures/si/. `deep-data` (optional) regenerates
# the pools from scratch. `deep-proper` redraws figS_dl_confidence from the 10k/10-seed run.
deep-data: dirs
	$(RUN) src/benchmark/deep/data.py                    # regenerate data/curated/deep/{X,y,t50}_{matched,natural}.npy

deep: dirs
	$(RUN) src/benchmark/deep/benchmark_dl.py            # results/benchmark/benchmark_dl.json + fig
	$(RUN) src/benchmark/deep/confidence_distribution.py # results/benchmark/checks/confidence_distribution.json + fig
	cp results/benchmark/fig_benchmark_dl.png figures/si/figS_benchmark_dl.png
	cp results/benchmark/checks/fig_confidence.png figures/si/figS_dl_confidence.png

deep-proper: dirs
	$(RUN) src/benchmark/deep/confidence_distribution.py proper   # 10k/10-seed; slow
	cp results/benchmark/checks/fig_confidence_proper.png figures/si/figS_dl_confidence.png

clean-figures:
	rm -f figures/main/*.png figures/si/*.png
