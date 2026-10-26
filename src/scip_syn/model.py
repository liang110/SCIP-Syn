"""Portable implementation of the current frozen-feature SCIP-Syn model."""
from dataclasses import dataclass
import torch
from torch import nn
from .layers import (ContextualTargetTower, CellLineTower, SymmetricDrugInteractionModule,
                     ContextAwareFusion, MultiTaskHead)


@dataclass(frozen=True)
class ModelConfig:
    molecular_dim: int = 768
    expression_dim: int = 2000
    hidden_dim: int = 256
    num_heads: int = 8
    dropout: float = 0.25
    context_hidden_dim: int = 256
    residual_scale: float = 0.5


def require_finite_output(module, inputs, output):
    for value in output if isinstance(output, (tuple, list)) else (output,):
        if isinstance(value, torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all():
            raise FloatingPointError(f'Nonfinite output: {type(module).__name__}')


class SCIPSyn(nn.Module):
    """Inputs are frozen molecular vectors, complete target ID sets and expression.

    No MoLFormer weights are included. Target ID 0 means PAD/missing information.
    The signature retains ignored cell IDs for old callers; no ID embedding exists.
    """
    def __init__(self, matrix, variant='full', config=None):
        super().__init__()
        self.config = config or ModelConfig()
        if variant not in ('full', 'target_input_off', 'static_query', 'static_target_context'):
            raise ValueError('Unknown matched-control variant')
        if matrix.ndim != 2 or matrix.shape[0] < 2 or not torch.isfinite(matrix).all() or matrix[0].ne(0).any():
            raise ValueError('Finite gene-pathway matrix with a zero PAD row required')
        self.variant = variant
        c = self.config
        self.num_targets = matrix.shape[0]
        self.register_buffer('target_information_enabled', torch.tensor(int(variant != 'target_input_off'), dtype=torch.long))
        self.molecular_projection = nn.Sequential(nn.Linear(c.molecular_dim, c.hidden_dim),
                                                  nn.LayerNorm(c.hidden_dim), nn.GELU(), nn.Dropout(c.dropout))
        self.target_tower = ContextualTargetTower(matrix.clone(), c.hidden_dim, c.dropout,
            c.expression_dim, c.context_hidden_dim, c.residual_scale)
        self.cell_line_tower = CellLineTower(c.expression_dim, c.hidden_dim, c.dropout)
        self.struct_interaction = SymmetricDrugInteractionModule(c.hidden_dim, c.dropout)
        self.target_interaction = SymmetricDrugInteractionModule(c.hidden_dim, c.dropout)
        controlled = variant in ('static_query', 'static_target_context')
        self.fusion = ContextAwareFusion(c.hidden_dim, c.num_heads, c.dropout, controlled, variant == 'static_query')
        self.multi_task_head = MultiTaskHead(c.hidden_dim, c.hidden_dim // 2, c.dropout)
        if controlled:
            self.register_buffer('target_context_is_static', torch.tensor(int(variant == 'static_target_context'), dtype=torch.long))
        for module in self.modules():
            if not list(module.children()):
                module.register_forward_hook(require_finite_output)

    def forward(self, mol_a, mol_b, targets_a, targets_b, expression, cell_ids=None, structure_mask=None):
        c = self.config
        if mol_a.ndim != 2 or mol_a.shape != mol_b.shape or mol_a.shape[1] != c.molecular_dim:
            raise ValueError('Molecular inputs must be equally shaped [batch, 768] arrays')
        if len(mol_a) == 0 or expression.shape != (len(mol_a), c.expression_dim):
            raise ValueError('Nonempty [batch, 2000] expression input required')
        if targets_a.ndim != 2 or len(targets_a) != len(mol_a) or targets_a.shape[1] == 0 or targets_b.shape != targets_a.shape:
            raise ValueError('Padded complete target-ID sets required')
        if not all(torch.isfinite(x).all() for x in (mol_a, mol_b, expression)):
            raise FloatingPointError('Nonfinite molecular/expression input')
        for ids in (targets_a, targets_b):
            if ids.dtype != torch.long or (ids < 0).any() or (ids >= self.num_targets).any():
                raise ValueError('Target IDs must be int64 inside the matrix vocabulary')
        za, zb = self.molecular_projection(mol_a), self.molecular_projection(mol_b)
        static_context = hasattr(self, 'target_context_is_static') and bool(self.target_context_is_static.item())
        target_expression = torch.zeros_like(expression) if static_context else expression
        ta = self.target_tower(targets_a * self.target_information_enabled, target_expression)
        tb = self.target_tower(targets_b * self.target_information_enabled, target_expression)
        context = self.cell_line_tower(expression)
        structural = self.struct_interaction(za, zb)
        if structure_mask is not None:
            structural = structural * structure_mask.to(structural.dtype).reshape(-1, 1)
        biological = self.target_interaction(ta, tb)
        return self.multi_task_head(self.fusion(torch.stack([structural, biological], dim=1), context))


def parameter_counts(model):
    return {'registered': sum(p.numel() for p in model.parameters()),
            'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad)}
