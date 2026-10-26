# Historical scientific-parity environment

These versions describe the original CPU source-equivalence audit, not the
recommended install environment. New installs use the modern tested environment
in the root `requirements-tested.txt` and PyTorch >=2.10.

- Python 3.11
- torch 2.1.0+cu118 (CPU-only execution in the audit)
- numpy 1.26.4
- pandas 3.0.0
- scipy 1.17.0

Historical results: 21 original core tests passed; Full and three matched controls
had identical ordered state keys/seeded weights, forward outputs and gradients;
AB/BA max difference was 0. Ten existing Full checkpoints each matched on 32
archived rows (320 rows, not all predictions). Complete stored prediction files
reproduced ten-fold PCC/Spearman/EF@1%/EF@5% within 1e-12.

All checkpoint/pickle reads were restricted to the owner's trusted files. The old
torch version is not a safe recommendation for untrusted file loading. No claim
of new-runtime bitwise-identical GPU retraining is made. The biological model
computation and frozen `configs/train.json` were preserved during software organization.
