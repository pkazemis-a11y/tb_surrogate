"""Clean surrogate-modeling utilities for diagrid tall buildings.

The repository exposes the dataset contract, a reference surrogate-model
implementation, and optimization utilities derived from the exploratory
research notebook.
"""

__version__ = "0.1.0"
__author__ = "Pooyan Kazemi"

from . import core, models, optimization, visualization

__all__ = [
    'core',
    'models',
    'optimization',
    'visualization',
]
