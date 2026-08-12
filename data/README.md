# Data

All raw sources are licensed CC BY 4.0 and are redistributed here with attribution.
Every file in `raw/` is a snapshot: the paper's numbers are pinned to the exact
vintages listed below, verified by `raw/CHECKSUMS.sha256`. Several sources update
continuously (HATCH, IEA, car sales), so a fresh download will not reproduce the
paper byte-for-byte; use the committed snapshots.

## Provenance

| File(s) in `raw/` | Source | License | Vintage / accessed | Citation |
|---|---|---|---|---|
| `technologies.csv` | HATCH: Historical Adoption of TeCHnology dataset (export) | CC BY 4.0 | ACCESS DATE TBC | Nemet et al. (2023), "Dataset on the adoption of historical technologies informs the scale-up of emerging carbon dioxide removal measures," *Communications Earth & Environment* 4:397. Data: [doi:10.5281/zenodo.8427842](https://zenodo.org/doi/10.5281/zenodo.8427842) |
| `IEA CCUS Projects Database 2026.xlsx` | IEA CCUS Projects Database | CC BY 4.0 | 2026 edition, ACCESS DATE TBC | IEA, *CCUS Projects Database*, IEA, Paris, <https://www.iea.org/data-and-statistics/data-product/ccus-projects-database>, Licence: CC BY 4.0 |
| `SoCDR-Edition-3/SoCDR-Edition-3-Chapter-5-G20-CDR-policy-database.csv`, `SoCDR-Edition-3/SoCDR-Edition-3-Chapter-7.csv` | State of Carbon Dioxide Removal, 3rd Edition (Ch. 5 G20 CDR-policy database; Ch. 7 realised novel-CDR removals by method) | CC BY 4.0 | Edition 3 (June 2026), ACCESS DATE TBC | Edwards et al. (2026), *The State of Carbon Dioxide Removal, 3rd Edition*. Data portal: <https://www.stateofcdr.org/data-portal/3rd-edition> |
| `all_carsales_monthly.csv` | Robbie Andrew, collected vehicle-registration statistics (BEV registrations by market) | CC BY 4.0 | Accessed December 2025 | Robbie Andrew, *Car sales data*, <https://robbieandrew.github.io/carsales/>. Note: Andrew flags some countries (e.g. Nepal) as indicative, lower-quality trade statistics |
| `owid_solar_wind_share.csv` | Our World in Data, solar and wind share of electricity generation | CC BY 4.0 | ACCESS DATE TBC | Our World in Data, compiled from Ember and the Energy Institute Statistical Review of World Energy, <https://ourworldindata.org/> |

License scope notes:

- IEA's CC BY 4.0 covers IEA-authored content only, not any component attributed
  to a third party. The CCUS Projects Database is IEA's own compilation.
- OWID processed datasets are CC BY 4.0; underlying sources (Ember, Energy
  Institute) require their own attribution, given above.
- HATCH aggregates published historical series (e.g. Mitchell); the Zenodo
  distribution as a whole is CC BY 4.0.

## Verifying snapshots

```bash
cd data/raw && sha256sum -c CHECKSUMS.sha256
```

## `curated/`

Derived inputs (per-tier `X_full.npy`, `lengths.npy`, `metadata.csv`,
`real_world_samples.pkl`) are regenerated from `raw/` by the curation pipeline
(`src/curation/`, see SI §S1: 10+ points, rising phase, onset trim, decline
truncation, PCHIP resampling). Binaries are not tracked in git; per-tier
`metadata.csv` and filter info are, as the audit trail of which series entered
each tier.
