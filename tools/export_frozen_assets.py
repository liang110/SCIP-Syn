"""One-way, local export from the TRUSTED historical environment to safe NPZ.

Run only against your own audited training directory: importing its common.py
executes local project code and reads trusted legacy pickles. This tool never
downloads files, contacts a service, or trains a model. Exported data are ignored
by Git and must not be submitted with the minimal source release.
"""
import argparse
import importlib.util
from pathlib import Path
import sys
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trusted-legacy-run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--i-trust-local-code-and-pickles', action='store_true')
    args = parser.parse_args()
    if not args.i_trust_local_code_and_pickles:
        parser.error('Explicit trust acknowledgement required; do not run on downloaded unknown directories')
    root, output = Path(args.trusted_legacy_run).resolve(), Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite feature assets')
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location('_trusted_frozen_common', root / 'common.py')
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    mol, cell, targets, matrix = common.assets()
    drugs, cells = sorted(mol), sorted(cell)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        np.savez_compressed(stream, drug_ids=np.array(drugs), cell_ids=np.array(cells),
            molecular=np.stack([mol[d] for d in drugs]), targets=np.stack([targets[d].numpy() for d in drugs]),
            expression=np.stack([cell[c] for c in cells]), gene_pathway=matrix.cpu().numpy())
    print(f'Exported {len(drugs)} drugs / {len(cells)} cells; no training executed')


if __name__ == '__main__':
    main()
