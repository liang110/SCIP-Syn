"""Only the executed molecular/biological/context fusion route.

Attribute names and layer order retain compatibility with the frozen model's
state_dict. Unused graph, motif, cell-ID and target-embedding routes are absent.
"""
import torch
from torch import nn


class KnowledgeAwareTargetTower(nn.Module):
    def __init__(self, matrix, hidden_dim, dropout):
        super().__init__()
        self.num_targets = matrix.shape[0]
        self.pathway_projection = nn.Linear(matrix.shape[0], matrix.shape[1], bias=False)
        self.pathway_projection.weight.data = matrix.t().clone()
        self.pathway_projection.weight.requires_grad_(False)
        self.encoder = nn.Sequential(
            nn.Linear(matrix.shape[1], hidden_dim * 2), nn.LayerNorm(hidden_dim * 2),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim))

    def forward(self, target_ids):
        multi_hot = torch.zeros(len(target_ids), self.num_targets, device=target_ids.device)
        multi_hot.scatter_add_(1, target_ids, torch.ones_like(target_ids, dtype=torch.float32))
        multi_hot[:, 0] = 0.0
        return self.encoder(self.pathway_projection(multi_hot))


class ContextualTargetTower(nn.Module):
    def __init__(self, matrix, hidden_dim, dropout, expression_dim, context_hidden_dim, residual_scale):
        super().__init__()
        self.residual_scale = residual_scale
        self.base_tower = KnowledgeAwareTargetTower(matrix, hidden_dim, dropout)
        self.context_encoder = nn.Sequential(
            nn.Linear(expression_dim, context_hidden_dim), nn.LayerNorm(context_hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(context_hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU())
        self.gamma = nn.Linear(hidden_dim, hidden_dim)
        self.beta = nn.Linear(hidden_dim, hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        for layer in (self.gamma, self.beta):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, target_ids, expression):
        base = self.base_tower(target_ids)
        context = self.context_encoder(expression.to(device=base.device, dtype=base.dtype))
        gamma, beta = torch.tanh(self.gamma(context)), self.beta(context)
        return self.dropout(self.out_norm(base * (1 + self.residual_scale * gamma)
                                         + self.residual_scale * beta))


class CellLineExpressionTower(nn.Module):
    def __init__(self, num_genes, hidden_dim, dropout):
        super().__init__()
        self.expression_encoder = nn.Sequential(
            nn.Linear(num_genes, 1024), nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout * 0.5))

    def forward(self, expression):
        return self.expression_encoder(expression)


class CellLineTower(nn.Module):
    def __init__(self, num_genes, hidden_dim, dropout):
        super().__init__()
        self.expression_tower = CellLineExpressionTower(num_genes, hidden_dim, dropout)

    def forward(self, expression):
        return self.expression_tower(expression)


class SymmetricDrugInteractionModule(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.interaction_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, 2 * hidden_dim), nn.LayerNorm(2 * hidden_dim, eps=1e-5),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-5), nn.GELU(), nn.Dropout(dropout))

    def forward(self, a, b):
        return self.interaction_mlp(torch.cat([a + b, a * b, torch.abs(a - b)], dim=-1))


class ContextAwareFusion(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, controlled=False, static_query=False):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, 2 * hidden_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(2 * hidden_dim, hidden_dim), nn.Dropout(dropout))
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-5)
        if controlled:
            self.register_buffer('query_is_static', torch.tensor(int(static_query), dtype=torch.long))

    def forward(self, pair_tokens, cell_repr):
        use_static = hasattr(self, 'query_is_static') and bool(self.query_is_static.item())
        query = (torch.zeros_like(cell_repr) if use_static else cell_repr).unsqueeze(1)
        out, _ = self.cross_attention(query=query, key=pair_tokens, value=pair_tokens)
        fused = self.norm1(out.squeeze(1) + cell_repr)
        return self.norm2(self.ffn(fused) + fused)


class MultiTaskHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        def head():
            return nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))
        self.regression_head = head()
        self.classification_head = head()

    def forward(self, x):
        shared = self.shared(x)
        return self.regression_head(shared), self.classification_head(shared)
