# Data and feature provenance

The manuscript data are organized as source tables, archived drug-exposure
assignments and execution code; see data/README.md and docs/REPRODUCTION.md.
Obtain third-party source resources through the links below under their own
licenses and access requirements. No third-party redistribution rights are implied.

## Endpoint

The frozen experiments reuse the **DrugComb v1.5** `synergy_loewe` summary field:

- Archive: https://zenodo.org/records/15235991
- File: `summary_v_1_5.csv`, 1,421,694,989 bytes.
- Raw MD5: `cd3e24758c10ffaea0124b6e80626c32`.
- Archived raw SHA256: `18c038d157a20d768f4a3f012628e5791ed628c784473777975eab7f75c185a0`.
- Frozen processed ALMANAC SHA256:
  `732bfad79d488eb7dd613d5aff4d370160bc16496ec3dca91411960651c2d252`.

These are upstream published Loewe summaries, not original ALMANAC ComboScore
and not locally refitted dose-response curves. Retained ALMANAC processing uses
numeric nonmissing scores, case-insensitive ALMANAC study selection, exclusion
outside [-100,100] (filtering, not clipping), cell-name/DepMap mapping and
expression availability. The archive contains 238,708 records / 103 drugs /
46 cells. Block/replicate rows remain separate during training; unordered
pair×cell averaging is an evaluation operation only.

## Feature resources and version drift

The verified earlier acquisition documentation is:
https://github.com/dan110liang110/SymContext-TP/blob/871376d3ac87234a271604b5f1e184581f685acb/DATA.md

It declares DepMap 25Q2 expression and provides a metadata route:
https://depmap.org/portal/data_page/?tab=allData
and https://ndownloader.figshare.com/files/35020903 . A declaration is not an
independently recovered byte match for every upstream expression source.

**Do not run the older public preprocessing defaults and call them identical to
the frozen experiment.** The frozen run skips the old activity-based drug filter
and prepares expression globally across 1,684 unlabeled ACH-cell references:
top 2,000 sample-variance genes and gene-wise Z scores using sample SD (ddof=1),
without the public script's SD epsilon. This is not fold-local preprocessing.
Use the archived feature ordering, scaler and split receipts for exact work.

The frozen internal molecular feature artifact (103×768) has SHA256
`2c5e4ac00e8619c1b8c1dc605b6cb78afeb5a7af3753c1ae311ba81a312a4447`.
Encoder identity is `ibm-research/MoLFormer-XL-both-10pct`, with the executed
attention-mask mean-pooled vectors. Encoder weights are not part of this repo.

## Safe NPZ input schema

| Key | Shape / meaning |
|---|---|
| `drug_ids` | Unicode array, D unique resolved drug identifiers |
| `molecular` | float32 [D,768], finite nonzero MoLFormer vectors |
| `targets` | int64 [D,T], all unique known IDs; 0 PAD/missing; T is observed width, not a cap |
| `cell_ids` | Unicode array, C unique cell identifiers |
| `expression` | float32 [C,2000], archived gene order and scaling |
| `gene_pathway` | float32 [3040,358], fixed map, row 0 all zero |

Readers use `allow_pickle=False`. Missing target information is represented as
zero target IDs, not evidence of absent biology. Missing molecular/expression
vectors are an error, not zero-imputed. The current internal target input covers
77/103 drugs; this package does not manufacture annotations for the other 26.

The optional `tools/export_frozen_assets.py` converts the owner's **trusted**
historical run to this schema. It imports that local run and reads trusted
pickles, requiring an explicit trust flag. Never point it at untrusted code or
downloaded pickle files. Exported NPZs remain outside version control.

## Splits

Preserve the historical inner fitting/validation assignment. Cold-start status is
defined against **complete outer train = fit ∪ inner validation**, not fit alone.
The published drug-exposure table and tools/create_study_splits.py recover outer
membership from the matching processed input. Historical Random training was row
based; removing repeated units from an evaluation subset does not retroactively
make it group-disjoint training. External transfer is retrospective and does
not create a fresh sealed lockbox.

The result-reproduction command uses the numerical source tables and does not
require trained weights. Training and frozen inference accept the feature arrays
and checkpoint files through their documented CLI input paths.

