"""Multi-objective building design optimization using NSGA-II.

Searches for Pareto-optimal tradeoffs between structural response and cost
within feasible design parameter ranges.
"""

import logging
from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination

from ..core.config import NSGAIIConfig, ALL_RESPONSE_COLUMNS


logger = logging.getLogger(__name__)


class BuildingDesignProblem(Problem):
    """
    Multi-objective optimization problem for building design.
    
    Minimizes multiple conflicting objectives:
    - Structural responses (accelerations, stresses, drift)
    - Economic cost (total mass)
    
    Subject to constraints on feasible parameter ranges.
    """
    
    def __init__(
        self,
        predictor,  # Surrogate model (trained MIMO-FNN)
        building_features: np.ndarray,
        objectives: List[str],
        objective_indices: List[int],
    ):
        """
        Initialize optimization problem.
        
        Args:
            predictor: Callable surrogate model that predicts responses.
            building_features: Template features for design parameters.
            objectives: Names of response variables to minimize.
            objective_indices: Indices of objectives in response vector.
        """
        self.predictor = predictor
        self.building_features = building_features
        self.objectives = objectives
        self.objective_indices = objective_indices
        self.num_objectives = len(objectives)
        
        # Define variable bounds for each design parameter
        # These represent feasible ranges based on engineering constraints
        self.num_variables = building_features.shape[1]
        
        # Infer practical bounds from the observed design space.
        lower_bounds = np.nanmin(building_features, axis=0)
        upper_bounds = np.nanmax(building_features, axis=0)
        
        super().__init__(
            n_var=self.num_variables,
            n_obj=self.num_objectives,
            n_constr=0,  # Can add constraints if needed
            type_var=np.float64,
            xl=lower_bounds,
            xu=upper_bounds,
        )
    
    def _evaluate(self, x: np.ndarray, out: Dict, *args, **kwargs):
        """
        Evaluate objective functions for population.
        
        Args:
            x: Population of design variables (pop_size x num_vars).
            out: Output dictionary to populate with objective values.
        """
        # Predict structural responses for all designs
        # This uses the trained surrogate model
        predictions = self.predictor(x)
        
        # Extract objective responses
        obj_values = predictions[:, self.objective_indices]
        
        # All objectives are minimized
        out['F'] = obj_values
        
        logger.debug(f"Evaluated {x.shape[0]} designs")


class DesignOptimizer:
    """
    Orchestrates multi-objective optimization using NSGA-II.
    
    Finds Pareto-optimal building designs that balance
    competing objectives like performance and cost.
    """
    
    def __init__(self, config: NSGAIIConfig):
        """
        Initialize optimizer.
        
        Args:
            config: NSGA-II configuration parameters.
        """
        self.config = config
        logger.info(f"DesignOptimizer initialized with {config.population_size} population")
    
    def optimize(
        self,
        problem: BuildingDesignProblem,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run NSGA-II optimization.
        
        NSGA-II is well-suited for this problem because:
        1. It handles multiple competing objectives naturally
        2. It maintains population diversity (Pareto frontier)
        3. It has fast non-dominated sorting
        4. It's proven effective for structural design optimization
        
        Args:
            problem: Optimization problem definition.
            
        Returns:
            Tuple of (optimal_designs, objective_values) where:
            - optimal_designs: Pareto-optimal design variables
            - objective_values: Corresponding objective function values
        """
        
        # Configure NSGA-II algorithm
        algorithm = NSGA2(
            pop_size=self.config.population_size,
            # Simulated binary crossover - good for continuous variables
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=self.config.crossover_prob, eta=15),
            # Polynomial mutation operator
            mutation=PM(eta=20),
            # Keep elite solutions
            eliminate_duplicates=True,
        )
        
        logger.info("Starting NSGA-II optimization...")
        logger.info(f"Objectives: {problem.objectives}")
        
        # Run optimization
        res = minimize(
            problem,
            algorithm,
            termination=get_termination(
                "n_gen",
                self.config.num_generations
            ),
            seed=self.config.random_seed,
            verbose=True,  # Show progress
        )
        
        logger.info(f"Optimization complete")
        logger.info(f"Found {res.F.shape[0]} Pareto-optimal solutions")
        
        return res.X, res.F
    
    def rank_solutions(
        self,
        designs: np.ndarray,
        objectives: np.ndarray,
        method: str = 'weighted_sum',
        weights: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Rank Pareto-optimal solutions for user selection.
        
        Args:
            designs: Optimal design variables.
            objectives: Corresponding objective values.
            method: Ranking method ('weighted_sum' or 'topsis').
            weights: Weights for each objective (if using weighted_sum).
            
        Returns:
            Tuple of (ranked_designs, ranked_objectives, ranks).
        """
        
        if weights is None:
            # Equal weighting by default
            weights = np.ones(objectives.shape[1]) / objectives.shape[1]
        
        if method == 'weighted_sum':
            # Normalize objectives to [0, 1] range
            obj_min = objectives.min(axis=0)
            obj_max = objectives.max(axis=0)
            obj_normalized = (objectives - obj_min) / (obj_max - obj_min + 1e-10)
            
            # Compute weighted score
            scores = obj_normalized @ weights
            
        elif method == 'topsis':
            # TOPSIS (Technique for Order Preference by Similarity)
            scores = self._topsis_scores(objectives)
        else:
            raise ValueError(f"Unknown ranking method: {method}")
        
        # Sort by score (ascending - lower is better)
        sort_idx = np.argsort(scores)
        
        ranked_designs = designs[sort_idx]
        ranked_objectives = objectives[sort_idx]
        ranks = np.arange(1, len(designs) + 1)
        
        logger.info(f"Solutions ranked using {method} method")
        
        return ranked_designs, ranked_objectives, ranks
    
    @staticmethod
    def _topsis_scores(objectives: np.ndarray) -> np.ndarray:
        """
        Compute TOPSIS scores for ranking.
        
        TOPSIS finds solutions closest to ideal and farthest from anti-ideal.
        
        Args:
            objectives: Objective matrix (solutions x objectives).
            
        Returns:
            TOPSIS scores for each solution.
        """
        # Normalize
        norm_obj = objectives / np.sqrt((objectives ** 2).sum(axis=0))
        
        # Ideal and anti-ideal solutions
        ideal = norm_obj.min(axis=0)  # Minimization problems
        anti_ideal = norm_obj.max(axis=0)
        
        # Separation measures
        s_ideal = np.sqrt(((norm_obj - ideal) ** 2).sum(axis=1))
        s_anti = np.sqrt(((norm_obj - anti_ideal) ** 2).sum(axis=1))
        
        # TOPSIS score: closeness to ideal solution
        scores = s_anti / (s_ideal + s_anti + 1e-10)
        
        return scores
    
    def export_solutions(
        self,
        solutions_df: pd.DataFrame,
        output_path: str,
    ):
        """
        Export optimized solutions to CSV.
        
        Args:
            solutions_df: DataFrame with design variables and objectives.
            output_path: Path to save CSV file.
        """
        solutions_df.to_csv(output_path, index=False)
        logger.info(f"Optimized solutions exported to {output_path}")


class NSGAIIOptimizerWrapper:
    """
    Pipeline-compatible wrapper for NSGA-II multi-objective optimization.
    
    Provides a simpler interface for the example pipeline that accepts
    a trainer (surrogate model) and runs optimization directly.
    """
    
    def __init__(
        self,
        surrogate_trainer,
        pop_size: int = 100,
        n_gen: int = 10,
        seed: int = 42,
    ):
        """Initialize wrapper with pipeline-style parameters.
        
        Args:
            surrogate_trainer: ModelTrainer instance with trained model
            pop_size: Population size for NSGA-II
            n_gen: Number of generations
            seed: Random seed
        """
        self.surrogate_trainer = surrogate_trainer
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.seed = seed
        
    def optimize(self) -> Dict:
        """
        Run optimization and return results.
        
        Returns:
            Dictionary with:
            - 'pareto_designs': Array of optimal design variables
            - 'pareto_objectives': Array of objective function values
        """
        # Build NSGA-II config from wrapper parameters
        config = NSGAIIConfig(
            population_size=self.pop_size,
            num_generations=self.n_gen,
            random_seed=self.seed,
        )
        if self.surrogate_trainer.model is None:
            raise ValueError("Surrogate trainer must be fit before optimization.")
        if self.surrogate_trainer.reference_ground_motion is None:
            raise ValueError("Ground-motion reference is unavailable. Fit the surrogate before optimization.")
        if self.surrogate_trainer.building_feature_bounds is None:
            raise ValueError("Building feature bounds are unavailable. Fit the surrogate before optimization.")

        objective_indices = [ALL_RESPONSE_COLUMNS.index(name) for name in config.objectives]
        lower_bounds, upper_bounds = self.surrogate_trainer.building_feature_bounds
        building_template = np.vstack([lower_bounds, upper_bounds])
        gm_reference = np.asarray(self.surrogate_trainer.reference_ground_motion, dtype=object)

        def predictor(candidate_buildings: np.ndarray) -> np.ndarray:
            gm_batch = np.repeat(gm_reference[None, :], candidate_buildings.shape[0], axis=0)
            return self.surrogate_trainer.predict(candidate_buildings, gm_batch)

        problem = BuildingDesignProblem(
            predictor=predictor,
            building_features=building_template,
            objectives=config.objectives,
            objective_indices=objective_indices,
        )

        optimizer = DesignOptimizer(config)
        logger.info("Running NSGA-II optimization via wrapper...")
        pareto_designs, pareto_objectives = optimizer.optimize(problem)

        return {
            'pareto_designs': pareto_designs,
            'pareto_objectives': pareto_objectives,
            'objective_names': config.objectives,
        }
