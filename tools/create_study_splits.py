"""Recover archived outer membership from the published drug-exposure table."""
import argparse,json,hashlib
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    source=Path(a.input);out=Path(a.output)
    if out.exists():raise FileExistsError('Use a new output directory')
    data=pd.read_csv(source)
    required={'drug1','drug2','cell_line_id','synergy_loewe'}
    if not required.issubset(data.columns) or data[list(required)].isna().any().any():
        raise ValueError('Input must be the processed, nonmissing study table')
    if len(data)!=238708:raise ValueError('Expected the 238,708-record ALMANAC study input')
    assignment=pd.read_csv(Path(__file__).resolve().parents[1]/'data/drug_exposure.csv')
    drugs=set(data.drug1)|set(data.drug2)
    for _,g in assignment.groupby('evaluation'):
        if drugs!=set(g.drug_id) or g.drug_id.duplicated().any():
            raise ValueError('Drug identities differ from the published study assignments')
    out.mkdir(parents=True)
    receipt={'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'assignments':[]}
    for evaluation,g in assignment.groupby('evaluation',sort=True):
        seen=set(g.loc[g.outer_train_exposed==1,'drug_id'])
        mask=data.drug1.isin(seen)&data.drug2.isin(seen)
        folder=out/f'fold_{evaluation}';folder.mkdir()
        for name,frame in [('train_cold_start',data[mask]),('test_cold_start',data[~mask])]:
            frame.to_csv(folder/f'{name}.csv',index=False)
        receipt['assignments'].append({'evaluation':int(evaluation),'train':int(mask.sum()),'test':int((~mask).sum())})
    (out/'membership_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
