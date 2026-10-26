# Reproducing SCIP-Syn results

This repository provides a compact result-reproduction route and the model training/inference implementation.

## Numerical reproduction

Run `python tools/reproduce_manuscript.py --output figure-output`. The command verifies checksums and reconstructs means and sample SD from archived numerical values. It performs no training or new inference.

Run `python tools/verify_candidate_replay.py --output candidate-replay` for independent CPU replay of 140 canonical score files (eight internal and six external methods). It checks candidate keys, strong-label assignments, budgets, PCC, Spearman, EF, counts, coverage and the published sensitivity summaries.

| Manuscript item | Published input |
| --- | --- |
| Table II, strict internal comparison | reproduce/frozen_fold_metrics.csv, internal / Selection-Unseen Both-Unseen |
| Tables S3c and S4a–S4b | internal and external populations in the same file |
| Tables S3a–S3b | reproduce/random_reference_summary.csv, reported aggregate means and sample SD |
| Figure 2A and its seven trajectories | reproduce/figure2_source.csv, panel a |
| Figure 2B–C | reproduce/figure2_source.csv, panels b and c |
| Figure 3 and selection counts | reproduce/strict_metrics.csv |
| Figure 4 and Tables S5b–S5c | reproduce/controls.csv and controls series in frozen_fold_metrics.csv |
| Fixed-pool uniform-selection reference | reproduce/internal_random_null.csv |
| Candidate-level replay | reproduce/canonical_scores.zip and canonical_manifest.json |
| Candidate N/K/B/H and distinct coverage | reproduce/candidate_pool_counts.csv and unique_coverage.csv |
| Paired main and subset contrasts | reproduce/paired_EF5_recomputed.csv |
| All four budget fractions | reproduce/budget_sensitivity.csv and budget_sensitivity_summary.csv |
| Molecular MLP external subsets | reproduce/MolecularMLP_external_all_populations.csv |
| Original fitting/validation row assignments | reproduce/inner_assignments.csv.gz and inner_assignment_manifest.json |
| Drug identity matching | reproduce/drug_identity_inventory.json and tools/identity_matching.py |

Figure 1 remains an implementation diagram. Figure 4 is retained in the manuscript; its matched numerical source values are supplied. The public drawing command produces Figures 2–3 only.

## Definitions

`series` identifies internal, external or control evaluations. `fold` is the archived evaluation identifier, used to pair methods within the same candidate pool. `population` names the eligibility subset. `model`, `metric` and `value` identify the corresponding scalar. Each estimand has ten archived values; SD is sample SD, not a confidence interval.

PCC and Spearman use continuous canonical responses. C-index is predictive concordance. EF@5% equals (hits/budget)/(strong labels/candidates); budget is ceil(0.05 × candidate count), and strong labels have canonical Loewe ≥10. Precision@5% is a proportion; manuscript tables convert it to percent. Sum Hits@5% for cumulative events and Budget@5% for cumulative slots. Macro precision averages ratios and differs from pooled precision. Repeated selections across external evaluations remain selection events.

`figure2_source.csv` contains the reported means and sample SD. Cross-regime points describe different fitting/candidate settings and are not paired causal effects. Negative mean-minus-SD endpoints must not be truncated. The three-dimensional bar depth separates PCC and Spearman only.

Figure 2(a) uses a base-2 PCC axis, with ticks labeled in original PCC units. Panels (b) and (c) retain linear correlation axes. Every panel uses lowercase parenthesized identifiers. The 5% budget applies to the entire eligible pool, not separately to each cell. Additional 1%, 2.5% and 10% budgets are sensitivity analyses; they do not replace the primary 5% analysis. See [EVIDENCE.md](EVIDENCE.md) for the scope of the results.

`controls.csv` contains evaluation, condition, PCC, EF@5%, hits and budget. Static query uses a fixed zero attention query while retaining the real-cell residual. Static target context supplies zero expression to the target conditioner. Target-input-off sets drug target IDs to zero. These controls were trained using the same selection protocol as Full.

## Input acquisition and partition identity

Follow DATA.md for upstream endpoint and feature sources. The data directory provides identity-only outer-training exposure assignments. `tools/create_study_splits.py` reconstructs outer membership from those assignments when supplied the matching processed input. It retains all input rows and their order. It does not invent an inner-validation assignment or reinterpret source-checkpoint evaluations as independent studies.

The released `inner_assignments.csv.gz` now preserves original fitting/validation positions relative to each archived outer-training CSV. The accompanying manifest pins the exact source bytes. Match those hashes before applying zero-based indices; duplicate records are mapped by occurrence. `configs/baseline_reference.json` records the actual Molecular MLP and ENV-Syn settings and source identifiers. No new split, tuning or model execution is performed by replay.

Result-summary reproduction uses the bundled CSV files. Training and frozen inference additionally use the input arrays and checkpoint paths specified by their CLI arguments. This distinction defines what each command reproduces without requiring unused logs or large upstream archives in the repository.

## Source identification

The evidence is identified as SCIP-Syn in reproduce/manifest.json. Hashes cover all supplied numerical inputs using the published LF text bytes or exact binary bytes; .gitattributes enforces the text convention. Hash verification remains enabled. Use the manuscript's immutable repository commit for citations. Software citation metadata are in CITATION.cff; no software DOI has been assigned.

