"""Canonical association and deterministic finite-budget retrieval estimands."""
import argparse
import math
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, hypergeom
from .data import write_json


def canonical_predictions(frame):
    required = ['drug1', 'drug2', 'cell_line_id', 'y_true_reg', 'y_pred_reg']
    if not set(required) <= set(frame):
        raise ValueError('Prediction columns missing')
    frame = frame[required].copy()
    if frame.isna().any().any() or len(frame) < 2:
        raise ValueError('Missing values or fewer than two prediction rows')
    for col in required[:3]:
        frame[col] = frame[col].astype(str).str.strip()
        if frame[col].str.len().eq(0).any():
            raise ValueError('Empty canonical identifier')
    frame[['drug1', 'drug2']] = np.sort(frame[['drug1', 'drug2']].to_numpy(dtype=str), axis=1)
    for col in required[3:]:
        frame[col] = pd.to_numeric(frame[col], errors='raise')
        if not np.isfinite(frame[col]).all():
            raise ValueError('Nonfinite prediction/response')
    return frame.groupby(required[:3], as_index=False, sort=True)[required[3:]].mean()


def evaluate(frame, outer_training_drugs=None, population='all', threshold=10.0):
    canonical = canonical_predictions(frame)
    if population != 'all':
        if outer_training_drugs is None:
            raise ValueError('Complete outer-training drug set required for exposure filtering')
        a = canonical.drug1.isin(outer_training_drugs)
        b = canonical.drug2.isin(outer_training_drugs)
        masks = {'both_unseen': ~a & ~b, 'one_unseen': a ^ b, 'both_seen': a & b}
        if population not in masks:
            raise ValueError('Unknown population')
        canonical = canonical[masks[population]].copy()
    if len(canonical) < 2:
        raise ValueError('Fewer than two eligible canonical units')
    y, p = canonical.y_true_reg.to_numpy(), canonical.y_pred_reg.to_numpy()
    constant = np.std(y) < 1e-12 or np.std(p) < 1e-12
    result = {'N': len(y), 'population': population,
              'PCC': None if constant else float(pearsonr(y, p).statistic),
              'Spearman': None if constant else float(spearmanr(y, p).statistic)}
    # Rank by score with an explicit canonical key for deterministic ties.
    ranked = canonical.sort_values(
        ['y_pred_reg', 'drug1', 'drug2', 'cell_line_id'],
        ascending=[False, True, True, True],
        kind='stable',
    )
    N, K = len(y), int((y >= threshold).sum())
    result['strong_labels'] = K
    for fraction in (0.01, 0.05):
        budget = int(math.ceil(fraction * N))
        hits = int((ranked.iloc[:budget].y_true_reg >= threshold).sum())
        prefix = f'top_{int(fraction * 100)}pct'
        result[prefix] = {'budget': budget, 'hits': hits, 'precision': hits / budget,
                          'EF': hits / budget / (K / N) if K else None,
                          'expected_random_hits': budget * K / N,
                          'random_hits_central95': [float(v) for v in hypergeom.ppf([.025, .975], N, K, budget)],
                          'conditional_random_tail': float(hypergeom.sf(hits - 1, N, K, budget))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--outer-train', help='Complete outer train, not only fit rows')
    parser.add_argument('--population', choices=['all', 'both_unseen', 'one_unseen', 'both_seen'], default='all')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    train = pd.read_csv(args.outer_train) if args.outer_train else None
    drugs = set(train.drug1.astype(str).str.strip()) | set(train.drug2.astype(str).str.strip()) if train is not None else None
    write_json(args.output, evaluate(pd.read_csv(args.predictions), drugs, args.population))


if __name__ == '__main__':
    main()
