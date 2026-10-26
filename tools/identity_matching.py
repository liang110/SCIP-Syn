import numpy as np

def identities_masks(f, outer, identity):
    alm = {x['drug_id']: x for x in identity['almanac']}
    ext = {x['drug_id']: x for x in identity['external']}

    def seen(drugs):
        names = {d.casefold() for d in drugs}
        smiles = {alm[d]['canonical_smiles'] for d in drugs}
        parents = {alm[d]['parent_identity'] for d in drugs}
        return {d: d.casefold() in names or x['canonical_smiles'] in smiles or x['parent_identity'] in parents for d, x in ext.items()}
    train = seen(set(outer.drug1) | set(outer.drug2))
    glob = seen(set(alm))
    n = (~f.drug1.map(train)).astype(int) + (~f.drug2.map(train)).astype(int)
    ng = (~f.drug1.map(glob)).astype(int) + (~f.drug2.map(glob)).astype(int)

    def parent_keys(a, lookup):
        p = a.drug1.map(lambda d: lookup[d]['parent_identity']).to_numpy()
        q = a.drug2.map(lambda d: lookup[d]['parent_identity']).to_numpy()
        return list(zip(np.minimum(p, q), np.maximum(p, q), a.cell_line_id))
    trained = set(parent_keys(outer, alm))
    overlap = np.array([k in trained for k in parent_keys(f, ext)])
    c = f.cell_line_id.isin(set(outer.cell_line_id)).to_numpy()
    named = (~f.drug1.isin(set(outer.drug1) | set(outer.drug2))).astype(int) + (~f.drug2.isin(set(outer.drug1) | set(outer.drug2))).astype(int)
    return {'External Overall': np.ones(len(f), bool), 'External Nonoverlap': ~overlap, 'External Both-Seen': n.eq(0).to_numpy(), 'External One-Unseen': n.eq(1).to_numpy(), 'External Both-Unseen': n.eq(2).to_numpy(), 'External Global-Drug-Unseen': ng.eq(2).to_numpy(), 'External Both-Unseen Cell-Seen': n.eq(2).to_numpy() & c, 'External Both-Unseen Cell-Unseen': n.eq(2).to_numpy() & ~c, 'External Name-Only Both-Unseen Sensitivity': named.eq(2).to_numpy()}
