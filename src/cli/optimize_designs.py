"""CLI entry point for design optimization using a trained surrogate model."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.core import (
    ALL_RESPONSE_COLUMNS,
    BUILDING_FEATURES,
    DataLoader,
    DataPaths,
    FeaturePreprocessor,
    GROUND_MOTION_FEATURES,
    NeuralNetworkConfig,
    NSGAIIConfig,
    PRIMARY_OPTIMIZATION_TARGETS,
    ResponseScaler,
)
from src.models import create_model
from src.optimization import BuildingDesignProblem, DesignOptimizer


def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """Configure logging for optimization runs."""
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    return root_logger


class SurrogatePredictor:
    """Apply stored preprocessing and model inference to candidate designs."""

    def __init__(
        self,
        model: torch.nn.Module,
        building_preprocessor: FeaturePreprocessor,
        gm_preprocessor: FeaturePreprocessor,
        response_scaler: ResponseScaler,
        gm_template: pd.DataFrame,
        device: str = 'cpu',
    ):
        self.model = model
        self.model.eval()
        self.building_preprocessor = building_preprocessor
        self.gm_preprocessor = gm_preprocessor
        self.response_scaler = response_scaler
        self.gm_template = gm_template
        self.device = device

    def predict(self, designs: np.ndarray) -> np.ndarray:
        """Predict responses for a population of candidate building designs."""
        building_df = pd.DataFrame(designs, columns=BUILDING_FEATURES)
        gm_df = pd.concat([self.gm_template] * len(building_df), ignore_index=True)

        building_processed = self.building_preprocessor.transform(building_df)
        gm_processed = self.gm_preprocessor.transform(gm_df)

        with torch.no_grad():
            building_tensor = torch.FloatTensor(building_processed).to(self.device)
            gm_tensor = torch.FloatTensor(gm_processed).to(self.device)
            predictions = self.model(building_tensor, gm_tensor)
            predictions_np = torch.cat(predictions, dim=1).cpu().numpy()

        return self.response_scaler.inverse_transform(predictions_np)

    def __call__(self, designs: np.ndarray) -> np.ndarray:
        return self.predict(designs)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for optimization."""
    parser = argparse.ArgumentParser(
        description='Optimize tall building designs using surrogate model predictions'
    )
    parser.add_argument(
        '--model', type=str, default=DataPaths.TRAINED_MODEL_PATH,
        help='Path to trained MIMO-FNN model weights'
    )
    parser.add_argument(
        '--building-prep', type=str, default=DataPaths.BUILDING_PREPROCESSOR_PATH,
        help='Path to building feature preprocessor'
    )
    parser.add_argument(
        '--gm-prep', type=str, default=DataPaths.GM_PREPROCESSOR_PATH,
        help='Path to ground-motion feature preprocessor'
    )
    parser.add_argument(
        '--response-scaler', type=str, default=DataPaths.RESPONSE_SCALER_PATH,
        help='Path to response scaler'
    )
    parser.add_argument(
        '--data-path', type=str, default=DataPaths.DATABASE_CSV,
        help='Path to CSV database'
    )
    parser.add_argument(
        '--population', type=int, default=100,
        help='NSGA-II population size'
    )
    parser.add_argument(
        '--generations', type=int, default=10,
        help='Number of optimization generations'
    )
    parser.add_argument(
        '--objectives', type=str, nargs='+',
        default=PRIMARY_OPTIMIZATION_TARGETS,
        help='Objectives to minimize (subset of all response variables)'
    )
    parser.add_argument(
        '--gm-strategy', type=str, choices=['mean', 'row'], default='mean',
        help='Strategy for providing ground-motion during optimization'
    )
    parser.add_argument(
        '--gm-row-index', type=int, default=0,
        help='Ground-motion row index (when gm-strategy=row)'
    )
    parser.add_argument(
        '--output', type=str, default='results/optimized_designs.csv',
        help='Output path for optimal designs CSV'
    )
    parser.add_argument(
        '--num-solutions', type=int, default=10,
        help='Number of top ranked solutions to save'
    )
    parser.add_argument(
        '--device', type=str, default='cpu', choices=['cpu', 'cuda'],
        help='Device for model inference'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable debug logging'
    )
    return parser.parse_args()


def build_gm_template(X_gm: pd.DataFrame, strategy: str, row_index: int) -> pd.DataFrame:
    """Build the fixed ground-motion template used during optimization."""
    if strategy == 'row':
        if row_index < 0 or row_index >= len(X_gm):
            raise ValueError(f'gm-row-index must be in [0, {len(X_gm) - 1}]')
        return X_gm.iloc[[row_index]].reset_index(drop=True)
    return pd.DataFrame([X_gm.mean(numeric_only=True)], columns=GROUND_MOTION_FEATURES)


def main() -> None:
    """Run NSGA-II over the building-design feature space."""
    args = parse_args()
    logger = setup_logging('DEBUG' if args.verbose else 'INFO')
    logger.info('Starting building design optimization')
    logger.info('Arguments: %s', args)

    try:
        loader = DataLoader(args.data_path)
        _ = loader.load()
        X_building, X_gm, _ = loader.get_features_and_responses()

        config = NeuralNetworkConfig(device=args.device)
        model = create_model(config=config, output_sizes=config.output_sizes, device=args.device)
        model.load_state_dict(torch.load(args.model, map_location=args.device))

        building_preprocessor = FeaturePreprocessor.load(args.building_prep)
        gm_preprocessor = FeaturePreprocessor.load(args.gm_prep)
        response_scaler = ResponseScaler.load(args.response_scaler)
        gm_template = build_gm_template(X_gm, args.gm_strategy, args.gm_row_index)

        predictor = SurrogatePredictor(
            model=model,
            building_preprocessor=building_preprocessor,
            gm_preprocessor=gm_preprocessor,
            response_scaler=response_scaler,
            gm_template=gm_template,
            device=args.device,
        )

        # Validate and map objective names to output indices
        objective_indices = []
        for objective in args.objectives:
            if objective not in ALL_RESPONSE_COLUMNS:
                raise ValueError(
                    f"Unknown objective '{objective}'. "
                    f"Available: {', '.join(ALL_RESPONSE_COLUMNS)}"
                )
            objective_indices.append(ALL_RESPONSE_COLUMNS.index(objective))

        problem = BuildingDesignProblem(
            predictor=predictor,
            building_features=X_building.to_numpy(),
            objectives=args.objectives,
            objective_indices=objective_indices,
        )

        optimizer = DesignOptimizer(
            NSGAIIConfig(
                population_size=args.population,
                num_generations=args.generations,
                objectives=args.objectives,
            )
        )
        optimal_designs, optimal_objectives = optimizer.optimize(problem)
        ranked_designs, ranked_objectives, ranks = optimizer.rank_solutions(optimal_designs, optimal_objectives)

        num_solutions = min(args.num_solutions, ranked_designs.shape[0])
        results_df = pd.DataFrame(ranked_designs[:num_solutions], columns=BUILDING_FEATURES)
        for idx, objective in enumerate(args.objectives):
            results_df[objective] = ranked_objectives[:num_solutions, idx]
        results_df.insert(0, 'Rank', ranks[:num_solutions])

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        logger.info('Saved %s ranked solutions to %s', num_solutions, output_path)
    except Exception as exc:
        logger.error('Optimization failed: %s', exc)
        sys.exit(1)
