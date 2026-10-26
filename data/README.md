# Study data information

The public data are intentionally compact: upstream source/version information, the drug-exposure assignments needed to identify archived cold-start populations, and numerical source files for the manuscript tables and figures.

- `drug_exposure.csv`: 1,030 rows; `evaluation` is the archived assignment identifier, `drug_id` is the resolved ALMANAC drug name, and `outer_train_exposed` is 1 when the drug appears in the complete outer training set and 0 otherwise. Every assignment covers the same 103 drugs.
- `../reproduce/figure2_source.csv`: plotting means and sample SD for Figure 2.
- `../reproduce/strict_metrics.csv`: verified strict Both-Unseen numerical results.
- `../reproduce/controls.csv`: matched input-control source values.
- `../reproduce/frozen_fold_metrics.csv`: the retained internal, external and control summaries.
- `../reproduce/manifest.json`: SHA256 checksums and included method sets.

Obtain the original DrugComb endpoint and feature resources through DATA.md. Use `tools/create_study_splits.py` with the matching processed ALMANAC file to recover the outer train/test membership. Supplementary tables summarize results; the machine-readable files retain the assignment identifiers needed for numerical pairing.
