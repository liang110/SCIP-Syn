"""Verify canonical score replay and paired evidence on CPU; no model execution."""
from pathlib import Path
import hashlib,json,zipfile,io,argparse
import numpy as np
import pandas as pd
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default='candidate-replay');a=p.parse_args()
 root=Path(__file__).resolve().parents[1];d=root/'reproduce';out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 m=json.loads((d/'manifest.json').read_text())
 for n,h in m['files'].items():assert hashlib.sha256((d/n).read_bytes()).hexdigest()==h,n
 meta=json.loads((d/'canonical_manifest.json').read_text());expected=pd.read_csv(d/'strict_metrics.csv',float_precision='round_trip')
 rows=[];coverage=[];sensitivity=[];selected={};hits={};candidates={};poolref={};label_delta=[]
 with zipfile.ZipFile(d/'canonical_scores.zip') as z:
  assert set(z.namelist())=={x['path'] for x in meta['files']}
  for item in meta['files']:
   b=z.read(item['path']);assert hashlib.sha256(b).hexdigest()==item['sha256']
   model,reg,file=item['path'].split('/');model=model.replace('_',' ');f=int(file.split('_')[1].split('.')[0])
   x=pd.read_csv(io.BytesIO(b),float_precision='round_trip');key=['drug_lo','drug_hi','cell_line_id'];assert len(x)==item['rows']
   assert not x.duplicated(key).any() and x.rank_order.is_unique
   ref=x.sort_values(key)[key+['y_true_raw']].reset_index(drop=True);pk=(reg,f)
   if pk in poolref:
    assert ref[key].equals(poolref[pk][key]);assert ref.y_true_raw.ge(10).equals(poolref[pk].y_true_raw.ge(10))
    delta=float(abs(ref.y_true_raw-poolref[pk].y_true_raw).max());assert delta<2e-5;label_delta.append(delta)
   else:poolref[pk]=ref
   rank=x.sort_values('rank_order');assert rank.y_pred_raw.is_monotonic_decreasing
   n=len(x);k=int(x.y_true_raw.ge(10).sum());budget=int(np.ceil(.05*n));chosen=rank.head(budget);h=int(chosen.y_true_raw.ge(10).sum())
   vals={'PCC':x.y_true_raw.corr(x.y_pred_raw),'Spearman':x.y_true_raw.corr(x.y_pred_raw,method='spearman'),'EF@5%':h/budget/(k/n),'Hits@5%':h,'Precision@5%':h/budget}
   for metric,value in vals.items():
    v=expected[(expected.regime==reg)&(expected.model==model)&(expected.fold==f)&(expected.metric==metric)].value
    assert len(v)==1 and abs(value-v.iloc[0])<1e-10,(reg,model,f,metric)
   rows.append(dict(regime=reg,model=model,evaluation_id=f,N=n,K=k,B=budget,H=h,EF5=vals['EF@5%']))
   ky=lambda frame:set(map(tuple,frame[key].astype(str).to_numpy()));mk=(reg,model)
   for obj,frame in [(candidates,x),(selected,chosen),(hits,chosen[chosen.y_true_raw>=10])]:obj.setdefault(mk,set()).update(ky(frame))
   for q in [.01,.025,.05,.10]:
    bq=int(np.ceil(q*n));hq=int(rank.head(bq).y_true_raw.ge(10).sum());sensitivity.append(dict(regime=reg,model=model,evaluation_id=f,budget_fraction=q,N=n,K=k,B=bq,H=hq,precision=hq/bq,EF=hq/bq/(k/n)))
 pools=pd.DataFrame(rows);published=pd.read_csv(d/'candidate_pool_counts.csv')
 assert pools[['regime','model','evaluation_id']].equals(published[['regime','model','evaluation_id']])
 assert np.allclose(pools[['N','K','B','H','EF5']],published[['N','K','B','H','EF5']],rtol=0,atol=1e-12)
 for (reg,model),sub in pools.groupby(['regime','model'],sort=False):
  mk=(reg,model);coverage.append(dict(regime=reg,model=model,unique_candidates=len(candidates[mk]),selection_events=int(sub.B.sum()),unique_selected=len(selected[mk]),hit_events=int(sub.H.sum()),unique_hits=len(hits[mk])))
 cov=pd.DataFrame(coverage);assert cov.equals(pd.read_csv(d/'unique_coverage.csv'))
 sensitivity=pd.DataFrame(sensitivity);ps=pd.read_csv(d/'budget_sensitivity.csv');assert sensitivity[['regime','model','evaluation_id','budget_fraction']].equals(ps[['regime','model','evaluation_id','budget_fraction']]);assert np.allclose(sensitivity.select_dtypes('number'),ps.select_dtypes('number'),rtol=0,atol=1e-12)
 frozen=pd.read_csv(d/'frozen_fold_metrics.csv');ext=frozen[frozen.series=='external'];contrasts=pd.read_csv(d/'paired_EF5_recomputed.csv')
 for row in contrasts.itertuples():
  source=expected[expected.regime=='ALMANAC'] if row.population=='Internal Both-Unseen' else ext[ext.population==row.population]
  pivot=source[source.metric=='EF@5%'].pivot(index='fold',columns='model',values='value');v=(pivot['SCIP-Syn']-pivot[row.baseline]).to_numpy();loo=(v.sum()-v)/9
  assert np.allclose([v.mean(),loo.min(),loo.max()],[row.delta,row.lofo_low,row.lofo_high],rtol=0,atol=1e-12)
 for name,frame in [('candidate_pool_counts',pools),('unique_coverage',cov),('budget_sensitivity',sensitivity)]:frame.to_csv(out/(name+'.csv'),index=False)
 report={'status':'PASS','canonical_files':len(rows),'candidate_rows':int(pools.N.sum()),'methods_internal':8,'methods_external':6,'same_candidate_keys_and_strong_labels':True,'max_historical_label_precision_delta':max(label_delta),'current_metrics_tolerance':1e-10,'paired_comparisons':len(contrasts),'new_training':False,'new_inference':False}
 (out/'acceptance.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
