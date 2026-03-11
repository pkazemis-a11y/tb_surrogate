"""
Surrogate modeling system for tall building design optimization.

This package provides complete ML pipeline for:
- Training neural network surrogate models
- Multi-objective design optimization
- Structural response prediction
"""

__version__ = "0.1.0"
__author__ = "Pooyan Kazemi"

from . import core, models, optimization

__all__ = [
    'core',
    'models',
    'optimization',
]
