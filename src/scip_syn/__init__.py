"""SCIP-Syn: the current frozen-feature model, without historical architectures."""
from .model import SCIPSyn, ModelConfig, parameter_counts

__all__ = ["SCIPSyn", "ModelConfig", "parameter_counts"]
__version__ = "0.2.0"
