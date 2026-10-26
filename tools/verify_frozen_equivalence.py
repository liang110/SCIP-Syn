"""CPU-only comparison with the owner's TRUSTED historical model and assets.

This is an opt-in source-compatibility check, not model training. The original
run must already exist locally with its own dependencies. Nothing is downloaded.
No absolute cluster paths, checkpoints or private data are committed here.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch
from scip_syn.model import SCIPSyn, parameter_counts
from scip_syn.data import Assets, MolecularPairs, forward_batch
from scip_syn.metrics import evaluate
from scip_syn.training import fixed_loss, scheduler


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trusted-legacy-run', required=True)
    parser.add_argument('--trusted-controls-model', required=True)
    parser.add_argument('--assets', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--i-trust-local-code-and-pickles', action='store_true')
    args = parser.parse_args()
    if not args.i_trust_local_code_and_pickles:
        parser.error('Explicit trust acknowledgement required')
    torch.set_num_threads(2)
    root = Path(args.trusted_legacy_run).resolve()
    sys.path.insert(0, str(root))
    import common
    original = load_module('_frozen_original_model', root / 'model.py')
    common.SOURCE = root
    common.CONFIG['variants'] = ['full', 'target_information_off', 'static_query', 'static_target_context']
    controlled = load_module('_frozen_control_model', Path(args.trusted_controls_model))
    assets = Assets(args.assets)
    report = {'training_runs': 0, 'device': 'cpu', 'torch': str(torch.__version__), 'variants': {}}
    torch.manual_seed(123)
    inputs = (torch.randn(4, 768), torch.randn(4, 768), torch.tensor([[1, 2, 0]] * 4),
              torch.tensor([[3, 4, 0]] * 4), torch.randn(4, 2000), torch.zeros(4, dtype=torch.long))
    for variant in ['full', 'target_input_off', 'static_query', 'static_target_context']:
        old_class = original.MoLFormerContextKG if variant in ('full', 'target_input_off') else controlled.MoLFormerContextKG
        old_variant = 'target_information_off' if variant == 'target_input_off' else variant
        torch.manual_seed(42)
        before = old_class(assets.matrix, old_variant)
        torch.manual_seed(42)
        after = SCIPSyn(assets.matrix, variant)
        old_state, new_state = before.state_dict(), after.state_dict()
        assert list(old_state) == list(new_state), (variant, set(old_state) ^ set(new_state))
        assert all(torch.equal(old_state[k], new_state[k]) for k in old_state), variant
        after.load_state_dict(old_state, strict=True)
        before.eval(); after.eval()
        with torch.no_grad():
            a, b = before(*inputs), after(*inputs)
            eval_delta = max(float((x - y).abs().max()) for x, y in zip(a, b))
            reverse = after(inputs[1], inputs[0], inputs[3], inputs[2], inputs[4], inputs[5])
            swap_delta = max(float((x - y).abs().max()) for x, y in zip(b, reverse))
        before.train(); after.train()
        torch.manual_seed(789); a = before(*inputs)
        torch.manual_seed(789); b = after(*inputs)
        train_delta = max(float((x - y).abs().max()) for x, y in zip(a, b))
        fixed_loss(a[0], torch.zeros(4), a[1], torch.zeros(4)).backward()
        fixed_loss(b[0], torch.zeros(4), b[1], torch.zeros(4)).backward()
        gradient_delta = max(float((p.grad - dict(after.named_parameters())[name].grad).abs().max())
                             for name, p in before.named_parameters() if p.grad is not None)
        assert eval_delta == train_delta == gradient_delta == swap_delta == 0.0, variant
        report['variants'][variant] = {'state_keys_and_initial_values_identical': True,
            'eval_max_abs': eval_delta, 'training_forward_max_abs': train_delta,
            'gradient_max_abs': gradient_delta, 'swap_max_abs': swap_delta, 'parameters': parameter_counts(after)}
    # Asset conversion and data loader parity on real archived fit rows.
    mol, cell, targets, matrix = common.assets()
    assert torch.equal(matrix, assets.matrix)
    for drug, i in assets.drug_index.items():
        assert np.array_equal(mol[drug], assets.molecular[i])
        assert np.array_equal(targets[drug].numpy(), assets.targets[i])
    for name, i in assets.cell_index.items():
        assert np.array_equal(cell[name], assets.expression[i])
    fit = common.read_frame(common.full_dir(0) / '_inner_split/fit_train.csv')
    stats = {'mean': float(fit.synergy_loewe.mean()), 'std': float(fit.synergy_loewe.std(ddof=1))}
    vocab = {c: i + 1 for i, c in enumerate(sorted(set(fit.cell_line_id)))}
    old_data = common.MolecularPairs(fit, mol, cell, targets, stats, vocab)
    new_data = MolecularPairs(fit, assets, stats)
    for i in [0, 1, 2, len(fit) - 1]:
        assert all(torch.equal(old_data[i][j], new_data[i][j]) for j in [0, 1, 2, 3, 4, 6, 7])
    report['real_assets_exactly_preserved'] = True
    report['real_row_input_and_target_parity'] = True
    # Existing checkpoints are read, never trained or selected again.
    checkpoint_report = []
    for fold in range(10):
        path = root / 'results/full' / f'fold_{fold}' / 'best_model.pt'
        cp = torch.load(path, map_location='cpu', weights_only=True)
        old = original.MoLFormerContextKG(matrix, 'full').eval()
        new = SCIPSyn(matrix).eval()
        old.load_state_dict(cp['model_state'], strict=True)
        new.load_state_dict(cp['model_state'], strict=True)
        test = common.read_frame(common.split_dir(fold) / 'test_cold_start.csv').iloc[:32]
        ds = MolecularPairs(test, assets, cp['target_stats'])
        batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=32)))
        with torch.no_grad():
            x = forward_batch(old, batch, 'cpu')[0]
            y = forward_batch(new, batch, 'cpu')[0]
        delta = float((x - y).abs().max())
        assert delta == 0.0
        checkpoint_report.append({'fold': fold, 'rows': len(test), 'max_abs': delta})
    report['ten_existing_checkpoints'] = checkpoint_report
    # Match the original aggregate and ranking implementation on existing files.
    analysis = load_module('_trusted_analysis_common', common.FROZEN / 'tools/analysis_common.py')
    metric_report = []
    for fold in range(10):
        import pandas as pd
        frame = pd.read_csv(root / 'results/full' / f'fold_{fold}' / 'predictions.csv')
        outer = common.read_frame(common.split_dir(fold) / 'train_cold_start.csv')
        drugs = common.drugset(outer)
        ours = evaluate(frame, drugs, 'both_unseen')
        legacy_frame = frame.rename(columns={'y_true_reg': 'y_true_raw', 'y_pred_reg': 'y_pred_raw'})
        canon = analysis.aggregate_pair_cell(legacy_frame)
        canon = canon[~canon.drug_lo.isin(drugs) & ~canon.drug_hi.isin(drugs)]
        legacy = analysis.regression_metrics(canon)
        legacy.update(analysis.topk_metrics(canon))
        for key in ['PCC', 'Spearman']:
            assert abs(ours[key] - legacy[key]) < 1e-12
        for q in [1, 5]:
            assert abs(ours[f'top_{q}pct']['EF'] - legacy[f'EF@{q}%']) < 1e-12
        metric_report.append({'fold': fold, 'N': ours['N'], 'PCC': ours['PCC'],
                              'EF1': ours['top_1pct']['EF'], 'EF5': ours['top_5pct']['EF']})
    report['ten_fold_frozen_metric_parity'] = metric_report
    report['status'] = 'PASS'
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
