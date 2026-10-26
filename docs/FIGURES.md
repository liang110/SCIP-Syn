# SCIP-Syn figures

| Figure | PNG | Content | Numerical source |
|---|---|---|---|
| Architecture | [architecture.png](architecture.png) | Molecular and biological pair evidence, cell query and cross-attention | Model implementation in `src/scip_syn/` |
| Figure 2 | [Figure2.png](figures/Figure2.png) | Cross-regime PCC trajectories and internal/external association comparisons | `reproduce/figure2_source.csv` |
| Figure 3 | [Figure3.png](figures/Figure3.png) | Fixed-budget enrichment and cumulative strong-label events | `reproduce/strict_metrics.csv` |
| Figure 4 | [Figure4.png](figures/Figure4.png) | Matched controls for cell query and biological inputs | `reproduce/controls.csv` |

The architecture PNG is the author-supplied image. All PNGs are copied without
altering their pixels or panel labels. Figures 2 and 3 can also be regenerated:

```bash
python tools/reproduce_manuscript.py --output figure-output
```

The Figure 4 PNG is the retained manuscript figure; its numerical control results
are supplied separately. See [REPRODUCTION.md](REPRODUCTION.md) for definitions.
