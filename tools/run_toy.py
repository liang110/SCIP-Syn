"""Generate synthetic assets and run a tiny CPU pipeline. NOT biological evidence."""
import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from scip_syn.data import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='demo-output')
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20261015)
    drugs = np.array([f'TOY-D{i}' for i in range(10)])
    cells = np.array([f'TOY-C{i}' for i in range(4)])
    targets = np.array([[2*i+1, 2*i+2, 0] for i in range(10)], dtype=np.int64)
    matrix = np.zeros((3040, 358), dtype=np.float32)
    matrix[np.arange(1, 21), np.arange(20) % 12] = 1
    assets = root/'synthetic_assets.npz'
    np.savez_compressed(assets, drug_ids=drugs, cell_ids=cells, targets=targets,
        molecular=rng.normal(size=(10, 768)).astype(np.float32),
        expression=rng.normal(size=(4, 2000)).astype(np.float32), gene_pathway=matrix)

    def rows(pairs):
        return pd.DataFrame([{'drug1': a, 'drug2': b, 'cell_line_id': c,
            'synergy_loewe': float(rng.normal(4, 12))} for a, b, c in pairs])
    seen_pairs = [(a, b, c) for a, b in itertools.combinations(drugs[:6], 2) for c in cells]
    fit = rows([seen_pairs[i] for i in rng.integers(0, len(seen_pairs), size=64)])
    val = rows([seen_pairs[i] for i in rng.integers(0, len(seen_pairs), size=16)])
    test = rows([(a, b, c) for a, b in itertools.combinations(drugs[6:], 2) for c in cells])
    paths = {'assets': assets}
    for name, frame in [('fit', fit), ('validation', val), ('test', test)]:
        paths[name] = root/f'{name}.csv'
        frame.to_csv(paths[name], index=False)
    outer = root/'complete_outer_train.csv'
    pd.concat([fit, val], ignore_index=True).to_csv(outer, index=False)
    manifest = root/'hashes.json'
    write_json(manifest, {k: sha256(v) for k, v in paths.items()})
    config = json.loads((Path(__file__).resolve().parents[1]/'configs/train.json').read_text())
    # Toy-only changes: two CPU epochs, small batch, no worker subprocesses.
    config.update(epochs=2, patience=2, warmup_epochs=0, batch_size=16, num_workers=0)
    config_path = root/'toy_config.json'
    write_json(config_path, config)

    def run(module, *flags):
        subprocess.run([sys.executable, '-m', module, *map(str, flags)], check=True)
    run('scip_syn.training', '--assets', assets, '--fit', paths['fit'], '--validation', paths['validation'],
        '--test', paths['test'], '--manifest', manifest, '--config', config_path,
        '--device', 'cpu', '--output', root/'fit_run')
    checkpoint = root/'fit_run/best_model.pt'
    run('scip_syn.predict', '--assets', assets, '--checkpoint', checkpoint,
        '--checkpoint-sha256', sha256(checkpoint), '--rows', paths['test'],
        '--device', 'cpu', '--output', root/'frozen_predictions.csv')
    run('scip_syn.metrics', '--predictions', root/'frozen_predictions.csv', '--outer-train', outer,
        '--population', 'both_unseen', '--output', root/'toy_metrics.json')
    audit = json.loads((root/'frozen_predictions.audit.json').read_text())
    write_json(root/'demo_summary.json', {'synthetic': True, 'biological_performance_evidence': False,
        'cpu_only': True, 'toy_epochs': 2, 'test_rows': len(test), 'swap_max_abs': audit['swap_max_abs'],
        'status': 'PASS', 'note': 'Random features and responses; never cite these as manuscript performance.'})
    print('PASS: synthetic train -> trusted-checkpoint inference -> cold-start metrics; not biological evidence.')


if __name__ == '__main__':
    main()
