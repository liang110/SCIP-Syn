"""Frozen inference; labels do not select or recalibrate a checkpoint."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import Assets, MolecularPairs, read_frame, forward_batch, write_json, sha256
from .model import SCIPSyn, ModelConfig


@torch.no_grad()
def predict(model, loader, stats, device='cpu', swap=False, standardized=False):
    model.eval()
    values, logits = [], []
    for batch in loader:
        reg, logit, _, _ = forward_batch(model, batch, device, swap=swap)
        values.append(reg.reshape(-1).cpu().numpy())
        logits.append(logit.reshape(-1).cpu().numpy().astype(float))
    values = np.concatenate(values)
    if not np.isfinite(values).all():
        raise FloatingPointError('Nonfinite predictions')
    return (values if standardized else values.astype(float) * stats['std'] + stats['mean']), np.concatenate(logits)


def load_model(path, assets, device, allow_new_feature_assets=False, expected_checkpoint_sha256=None):
    # weights_only avoids the arbitrary-code pickle fallback. Never retry with False.
    if not expected_checkpoint_sha256 or sha256(path) != expected_checkpoint_sha256:
        raise ValueError('Trusted expected checkpoint SHA256 required before deserialization')
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    variant = checkpoint['variant']
    if variant == 'target_information_off':
        variant = 'target_input_off'
    model = SCIPSyn(assets.matrix, variant, ModelConfig(**checkpoint.get('model_config', {}))).to(device)
    frozen_mapping = checkpoint['model_state']['target_tower.base_tower.pathway_projection.weight']
    if not torch.equal(frozen_mapping.cpu(), assets.matrix.t().cpu()):
        raise ValueError('Checkpoint and asset gene-pathway mappings differ')
    model.load_state_dict(checkpoint['model_state'], strict=True)
    expected = int(variant != 'target_input_off')
    if model.target_information_enabled.item() != expected:
        raise ValueError('Checkpoint control buffer disagrees with variant')
    for name, active in [('target_context_is_static', variant == 'static_target_context')]:
        if hasattr(model, name) and getattr(model, name).item() != int(active):
            raise ValueError('Checkpoint context-control buffer disagrees with variant')
    if hasattr(model.fusion, 'query_is_static') and model.fusion.query_is_static.item() != int(variant == 'static_query'):
        raise ValueError('Checkpoint query-control buffer disagrees with variant')
    if not allow_new_feature_assets and checkpoint.get('assets_sha256') and sha256(assets.path) != checkpoint['assets_sha256']:
        raise ValueError('Checkpoint feature asset hash mismatch')
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--checkpoint-sha256', required=True, help='Expected hash obtained from the trusted checkpoint owner, not an untrusted download')
    parser.add_argument('--rows', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--allow-new-feature-assets', action='store_true', help='Explicit opt-in for external-study assets; frozen gene/pathway and expression ordering must remain identical')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Refusing to overwrite predictions')
    assets = Assets(args.assets)
    assets.path = args.assets
    model, checkpoint = load_model(args.checkpoint, assets, args.device, args.allow_new_feature_assets, args.checkpoint_sha256)
    frame = read_frame(args.rows, require_response=False)
    stats = checkpoint['target_stats']
    if not np.isfinite([stats['mean'], stats['std']]).all() or stats['std'] <= 0:
        raise ValueError('Invalid fit-only target statistics')
    loader = DataLoader(MolecularPairs(frame, assets, stats, require_response=False), batch_size=args.batch_size)
    scores, _ = predict(model, loader, stats, args.device)
    reverse, _ = predict(model, loader, stats, args.device, swap=True)
    if not np.allclose(scores, reverse, rtol=1e-5, atol=1e-4):
        raise ValueError('AB/BA audit failed')
    result = frame[['drug1', 'drug2', 'cell_line_id']].copy()
    if 'synergy_loewe' in frame:
        result['y_true_reg'] = frame.synergy_loewe
    result['y_pred_reg'], result['y_pred_reg_swapped'] = scores, reverse
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    write_json(output.with_suffix('.audit.json'), {'checkpoint_sha256': sha256(args.checkpoint),
        'assets_sha256': sha256(args.assets), 'rows_sha256': sha256(args.rows),
        'swap_max_abs': float(np.max(np.abs(scores - reverse))), 'rows': len(frame)})


if __name__ == '__main__':
    main()
