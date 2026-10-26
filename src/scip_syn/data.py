"""Explicit assets and archived split readers; never generates new splits."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def verify_manifest(manifest_path, named_paths):
    """Require explicit byte hashes for every input, independent of its location."""
    expected = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    actual = {name: sha256(path) for name, path in named_paths.items()}
    for name, digest in actual.items():
        if expected.get(name) != digest:
            raise ValueError(f'Missing or mismatched input SHA256: {name}')
    return actual


def read_frame(path, require_response=True):
    frame = pd.read_csv(path, dtype={'drug1': str, 'drug2': str, 'cell_line_id': str})
    for col in ('drug1', 'drug2', 'cell_line_id'):
        if col not in frame or frame[col].isna().any():
            raise ValueError(f'Missing identifiers: {col}')
        frame[col] = frame[col].str.strip()
        if frame[col].str.len().eq(0).any():
            raise ValueError(f'Empty identifiers: {col}')
    if len(frame) == 0:
        raise ValueError('Empty split')
    if require_response:
        frame['synergy_loewe'] = pd.to_numeric(frame['synergy_loewe'], errors='raise')
        if not np.isfinite(frame.synergy_loewe).all():
            raise ValueError('Nonfinite response')
    return frame


class Assets:
    def __init__(self, path):
        self.path = Path(path)
        with np.load(path, allow_pickle=False) as z:
            self.drugs = [str(v).strip() for v in z['drug_ids']]
            self.cells = [str(v).strip() for v in z['cell_ids']]
            self.molecular = np.array(z['molecular'], dtype=np.float32)
            self.expression = np.array(z['expression'], dtype=np.float32)
            raw_targets = z['targets']
            if raw_targets.dtype.kind not in 'iu':
                raise ValueError('Integer target-ID array required')
            self.targets = np.array(raw_targets, dtype=np.int64)
            self.matrix = torch.from_numpy(np.array(z['gene_pathway'], dtype=np.float32))
        if len(set(self.drugs)) != len(self.drugs) or len(set(self.cells)) != len(self.cells):
            raise ValueError('Duplicate asset identifiers')
        if not self.drugs or not self.cells or any(not x for x in self.drugs + self.cells):
            raise ValueError('Empty asset identifiers')
        if self.molecular.shape != (len(self.drugs), 768) or self.expression.shape != (len(self.cells), 2000):
            raise ValueError('Frozen feature dimensions must be 768 molecular / 2000 expression')
        if self.targets.ndim != 2 or self.targets.shape[0] != len(self.drugs) or self.targets.shape[1] == 0:
            raise ValueError('Complete padded target IDs required; do not truncate')
        if self.matrix.shape != (3040, 358) or self.matrix[0].ne(0).any():
            raise ValueError('Frozen gene-pathway vocabulary must be 3040 x 358 with PAD row zero')
        if not all(np.isfinite(x).all() for x in (self.molecular, self.expression, self.matrix.numpy())):
            raise ValueError('Nonfinite asset values')
        if (np.linalg.norm(self.molecular, axis=1) == 0).any():
            raise ValueError('Missing molecular features cannot silently become zero vectors')
        if self.targets.min() < 0 or self.targets.max() >= 3040:
            raise ValueError('Target IDs outside frozen vocabulary')
        for ids in self.targets:
            known = ids[ids != 0]
            if len(np.unique(known)) != len(known):
                raise ValueError('Targets must be unique sets, not repeated ID counts')
        self.drug_index = {v: i for i, v in enumerate(self.drugs)}
        self.cell_index = {v: i for i, v in enumerate(self.cells)}


class MolecularPairs(Dataset):
    def __init__(self, frame, assets, stats, require_response=True):
        frame = frame.copy()
        frame[['drug1', 'drug2']] = np.sort(frame[['drug1', 'drug2']].to_numpy(dtype=str), axis=1)
        drugs = sorted(set(frame.drug1) | set(frame.drug2))
        cells = sorted(set(frame.cell_line_id))
        if not set(drugs) <= set(assets.drug_index) or not set(cells) <= set(assets.cell_index):
            raise ValueError('Missing molecular/target/expression coverage for requested rows')
        dm, cm = {d: i for i, d in enumerate(drugs)}, {c: i for i, c in enumerate(cells)}
        self.mols = torch.from_numpy(np.stack([assets.molecular[assets.drug_index[d]] for d in drugs]))
        self.targets = torch.from_numpy(np.stack([assets.targets[assets.drug_index[d]] for d in drugs]))
        self.cells = torch.from_numpy(np.stack([assets.expression[assets.cell_index[c]] for c in cells]))
        self.indices = torch.tensor(list(zip(frame.drug1.map(dm), frame.drug2.map(dm), frame.cell_line_id.map(cm))))
        raw = frame.synergy_loewe.to_numpy(float) if require_response else np.zeros(len(frame))
        self.y = torch.from_numpy(((raw - stats['mean']) / stats['std']).astype(np.float32))
        self.cls = torch.from_numpy((raw >= 0).astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        a, b, c = self.indices[i]
        return (self.mols[a], self.mols[b], self.targets[a], self.targets[b],
                self.cells[c], torch.tensor(0), self.y[i], self.cls[i])


def forward_batch(model, batch, device, swap=False, structure_mask=None):
    a, b, ta, tb, cell, cid, y, cls = [v.to(device) for v in batch]
    if swap:
        a, b, ta, tb = b, a, tb, ta
    reg, logit = model(a, b, ta, tb, cell, cid, structure_mask=structure_mask)
    return reg, logit, y, cls
