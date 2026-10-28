# SCIP-Syn evidence scope

This snapshot retains eight internal methods, seven Random-to-cold-start trajectories and six frozen external methods. Molecular MLP external results are recovered from the completed frozen inference, not new training. Architecture, training parameters, checkpoints and the primary 5% budget are unchanged.

The canonical-score package contains only unordered drug identifiers, cell identifiers, observed labels, model scores and archived ordering. These are public-screen-derived research results, not clinical participant data. MIT covers source code; it does not relicense DrugComb, DepMap, target resources or pretrained weights. Upstream inputs remain subject to the terms in DATA.md. No raw feature arrays, weights, credentials, user directories or cluster logs are included.

## Interpretation

- Reference internal EF@5% is 1.4959 for SCIP-Syn and 1.4021 for Molecular MLP. Paired differences vary across pools.
- Frozen O'Neil EF@5% is 1.3483 versus 1.0443. The paired difference is +0.3040; deleting any one matched evaluation leaves a positive mean difference. LOFO is descriptive, not a confidence interval.
- O'Neil events are 286 versus 221 in 2,519 slots. Distinct strong-label pair-by-cell coverage is 191 versus 128. Coverage across repeated selections is not a single fixed-budget precision estimate.
- Budget sensitivity retains every method at 1%, 2.5%, 5% and 10%, including weaker small-budget internal results.

## Implementation boundaries

Molecular MLP uses molecular and expression features, without target/pathway input. Its 396,757 parameters do not capacity-match SCIP-Syn. ENV-Syn uses its explicit continuous-regression path. `configs/baseline_reference.json` records their actual settings and provenance hashes, without host paths. These comparisons do not isolate the attention operator.

`inner_assignments.csv.gz` gives original fitting/validation membership as zero-based row indices in the archived outer-training CSV. Repeated rows are matched by occurrence. Its manifest pins the source bytes; it does not silently resplit data or reconstruct labels from hashes.

Figures 2–3 use lowercase parenthesized panel labels. Figure 2(a) explicitly uses a base-2 PCC axis; (b), (c) retain 3D correlation bars. Figure 4 is retained at its original raster resolution; label edits do not create a high-resolution original.