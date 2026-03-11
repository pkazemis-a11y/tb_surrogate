"""Optimization module for multi-objective design optimization."""

from .nsga2 import DesignOptimizer, BuildingDesignProblem, NSGAIIOptimizerWrapper

# Alias for backward compatibility with example pipeline
NsGAIIOptimizer = NSGAIIOptimizerWrapper

__all__ = [
    'DesignOptimizer',
    'NsGAIIOptimizer',
    'NSGAIIOptimizerWrapper',
    'BuildingDesignProblem',
]
