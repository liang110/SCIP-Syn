"""Reproduce versioned numerical summaries and Figures 2–3 without training."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frozen(root):
    folder = Path(root)/'reproduce'
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        if digest(folder/name) != expected:
            raise ValueError(f'Frozen source data changed: {name}')
    frame = pd.read_csv(folder/'frozen_fold_metrics.csv', float_precision='round_trip')
    keys = ['series','population','model','metric']
    groups = frame.groupby(keys)
    if frame.duplicated(keys+['fold']).any() or not groups.size().eq(10).all():
        raise ValueError('Exactly ten distinct source-fold values per estimand required')
    if not np.isfinite(frame.value).all():
        raise ValueError('Nonfinite frozen metric')
    for target in manifest['expected_means']:
        sub = frame
        for key in keys:
            sub = sub[sub[key].eq(target[key])]
        if abs(sub.value.mean()-target['value']) > 1e-12:
            raise ValueError('Frozen mean acceptance check failed')
    null = pd.read_csv(folder/'internal_random_null.csv')
    if len(null)!=10 or not null.fold.is_unique:
        raise ValueError('Ten conditional-null rows required')
    return frame, null, manifest


def contrasts(controls):
    rows=[]
    keys=['population','metric']
    for (population, metric), group in controls.groupby(keys):
        full=group[group.model.eq('SCIP-Syn')].sort_values('fold').value.to_numpy()
        for control, sub in group[~group.model.eq('SCIP-Syn')].groupby('model'):
            delta=full-sub.sort_values('fold').value.to_numpy()
            loo=np.array([np.delete(delta,i).mean() for i in range(10)])
            rows.append(dict(population=population,metric=metric,control=control,
                mean_delta=float(delta.mean()),sample_sd=float(delta.std(ddof=1)),
                wins=int((delta>1e-12).sum()),ties=int((np.abs(delta)<=1e-12).sum()),
                losses=int((delta < -1e-12).sum()),leave_one_fold_min=float(loo.min()),
                leave_one_fold_max=float(loo.max())))
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='figure-output')
    parser.add_argument('--tables-only',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    frame,null,manifest=load_frozen(root)
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    summary=frame.groupby(['series','population','model','metric']).value.agg(['mean','std','count']).reset_index()
    summary.to_csv(output/'frozen_summary.csv',index=False)
    for series in ['trajectory','internal','external','controls']:
        summary[summary.series.eq(series)].to_csv(output/f'{series}_summary.csv',index=False)
    paired=contrasts(frame[frame.series.eq('controls')])
    paired.to_csv(output/'paired_controls.csv',index=False)
    if not args.tables_only:
        from manuscript_panels import draw
        tables={k:frame[frame.series.eq(k)].copy() for k in ['trajectory','internal','external','controls']}
        tables.update(random_null=null,control_contrasts=paired)
        draw(tables,output)
    report={'status':'PASS','source_rows':len(frame),'frozen_means_checked':len(manifest['expected_means']),
        'summary_rows':len(summary),'training_runs':0,'new_inference_runs':0,
        'figures':[] if args.tables_only else [2,3],
        'scope':'Numerical source-table reconstruction and deterministic plotting.',
        'source_hashes':manifest['files']}
    (output/'acceptance.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()

