"""Single-fold runner using supplied frozen fit/validation/test CSVs only."""
import argparse
import json
import math
from pathlib import Path
import random
import numpy as np
from scipy.stats import pearsonr
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .data import Assets, MolecularPairs, read_frame, forward_batch, verify_manifest, sha256, write_json
from .model import SCIPSyn, ModelConfig, parameter_counts
from .predict import predict


def fixed_loss(reg, y, logit, cls):
    return 5.0 * F.mse_loss(reg.squeeze(), y) + F.binary_cross_entropy_with_logits(logit.squeeze(), cls)


def scheduler(optimizer, warmup_steps, total_steps):
    def factor(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('assets', 'fit', 'validation', 'test', 'manifest', 'config', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--variant', choices=['full', 'target_input_off', 'static_query', 'static_target_context'], default='full')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if cfg['classification_threshold'] != 0 or cfg['reg_weight'] != 5 or cfg['cls_weight'] != 1 or cfg['modality_dropout_prob'] != 0 or cfg['target_std_ddof'] != 1:
        raise ValueError('This minimal runner implements the frozen objective and zero structural masking only')
    paths = {name: getattr(args, name) for name in ('assets', 'fit', 'validation', 'test')}
    hashes = verify_manifest(args.manifest, paths)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    fit, val = read_frame(args.fit), read_frame(args.validation)
    stats = {'mean': float(fit.synergy_loewe.mean()), 'std': float(fit.synergy_loewe.std(ddof=1))}
    if not np.isfinite(list(stats.values())).all() or stats['std'] <= 0:
        raise ValueError('Fit-only target SD must be finite and positive')
    if len(fit) % cfg['batch_size'] == 1:
        raise ValueError('One-row training batches are incompatible with the frozen BatchNorm/squeeze route')
    assets = Assets(args.assets)
    options = dict(batch_size=cfg['batch_size'], num_workers=cfg['num_workers'])
    train_loader = DataLoader(MolecularPairs(fit, assets, stats), shuffle=True, **options)
    val_loader = DataLoader(MolecularPairs(val, assets, stats), shuffle=False, **options)
    seed_all(cfg['seed'])
    model = SCIPSyn(assets.matrix, args.variant).to(args.device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    schedule = scheduler(optimizer, cfg['warmup_epochs'] * len(train_loader), cfg['epochs'] * len(train_loader))
    torch.save(model.state_dict(), output / 'initial_state.pt')
    write_json(output / 'run_manifest.json', {'config': cfg, 'model_config': vars(ModelConfig()),
        'variant': args.variant, 'hashes': hashes, 'target_stats': stats, 'parameter_counts': parameter_counts(model),
        'torch': str(torch.__version__), 'new_split_generation': False, 'outer_test_checkpoint_selection': False})
    best, stale, best_epoch = -float('inf'), 0, None
    for epoch in range(cfg['epochs']):
        model.train()
        total, steps = 0.0, 0
        for batch in train_loader:
            # Retain the historical RNG draw even though no structural pair is masked.
            mask = torch.rand(len(batch[-1]), device=args.device) < 1.0
            reg, logit, y, cls = forward_batch(model, batch, args.device, structure_mask=mask)
            loss = fixed_loss(reg, y, logit, cls)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss; no silently skipped batches')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
            optimizer.step()
            schedule.step()
            total += float(loss.detach())
            steps += 1
        values, _ = predict(model, val_loader, stats, args.device, standardized=True)
        pcc = float(pearsonr(val_loader.dataset.y.numpy(), values).statistic)
        if not np.isfinite(pcc):
            raise ValueError('Undefined inner PCC; no fallback selection metric')
        if pcc > best:
            best, stale, best_epoch = pcc, 0, epoch + 1
            torch.save({'model_state': model.state_dict(), 'variant': args.variant, 'model_config': vars(ModelConfig()),
                'target_stats': stats, 'best_epoch': best_epoch, 'inner_val_pcc': best,
                'assets_sha256': hashes['assets'], 'input_hashes': hashes}, output / 'best_model.pt')
        else:
            stale += 1
        record = {'epoch': epoch + 1, 'train_loss': total / steps, 'inner_val_pcc': pcc,
                  'best_epoch': best_epoch, 'lr': optimizer.param_groups[0]['lr']}
        with (output / 'epochs.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        print(json.dumps(record), flush=True)
        if stale >= cfg['patience']:
            break
    checkpoint = torch.load(output / 'best_model.pt', map_location=args.device, weights_only=True)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    # The test response is first read after checkpoint selection has finished.
    test = read_frame(args.test)
    loader = DataLoader(MolecularPairs(test, assets, stats), shuffle=False, **options)
    scores, _ = predict(model, loader, stats, args.device)
    reverse, _ = predict(model, loader, stats, args.device, swap=True)
    if not np.allclose(scores, reverse, rtol=1e-5, atol=1e-4):
        raise ValueError('AB/BA audit failed')
    predictions = test[['drug1', 'drug2', 'cell_line_id']].copy()
    predictions.insert(0, 'row_index', np.arange(len(test)))
    predictions['y_true_reg'], predictions['y_pred_reg'] = test.synergy_loewe.to_numpy(float), scores
    predictions['y_pred_reg_swapped'] = reverse
    predictions.to_csv(output / 'predictions.csv', index=False)
    write_json(output / 'complete.json', {'best_epoch': best_epoch, 'swap_max_abs': float(np.max(np.abs(scores - reverse))),
        'predictions_sha256': sha256(output / 'predictions.csv'), 'test_used_for_selection': False})


if __name__ == '__main__':
    main()
