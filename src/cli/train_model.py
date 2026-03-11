"""CLI entry point for surrogate-model training."""

import argparse
import logging
import sys
from pathlib import Path

from src.core import (
    BUILDING_FEATURES,
    DataLoader,
    DataPaths,
    FeaturePreprocessor,
    NeuralNetworkConfig,
    ResponseScaler,
)
from src.models import ModelTrainer, create_model


def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """Configure console logging for training runs."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_date_format = '%Y-%m-%d %H:%M:%S'

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    formatter = logging.Formatter(log_format, log_date_format)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    return root_logger


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description='Train the surrogate model for building design optimization'
    )
    parser.add_argument('--data-path', type=str, required=True, help='Path to CSV database file')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--hidden-size', type=int, default=512, help='Input branch hidden size')
    parser.add_argument('--dropout-rate', type=float, default=0.3, help='Dropout rate for shared layers')
    parser.add_argument('--k-folds', type=int, default=2, help='Number of CV folds')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Training device')
    parser.add_argument('--output-dir', type=str, default='models', help='Directory to save trained artifacts')
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')
    return parser.parse_args()


def main() -> None:
    """Run the end-to-end training workflow."""
    args = parse_args()
    logger = setup_logging('DEBUG' if args.verbose else 'INFO')
    logger.info('Starting surrogate-model training')
    logger.info('Arguments: %s', args)

    try:
        loader = DataLoader(args.data_path)
        _ = loader.load()
        X_building, X_gm, y = loader.get_features_and_responses()

        config = NeuralNetworkConfig(
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            k_fold_splits=args.k_folds,
            input_branch_hidden_size=args.hidden_size,
            dropout_rate_1=args.dropout_rate,
            dropout_rate_2=args.dropout_rate,
            device=args.device,
        )

        model = create_model(config=config, output_sizes=config.output_sizes, device=args.device)
        trainer = ModelTrainer(config=config, model=model, device=args.device)
        trainer.train_with_cross_validation(X_building, X_gm, y)

        building_preprocessor = FeaturePreprocessor()
        gm_preprocessor = FeaturePreprocessor()
        response_scaler = ResponseScaler(scaler_type='maxabs')

        X_building_processed = building_preprocessor.fit_transform(X_building)
        X_gm_processed = gm_preprocessor.fit_transform(X_gm)
        y_scaled = response_scaler.fit_transform(y.to_numpy())

        trainer.train_final_model(X_building_processed, X_gm_processed, y_scaled)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / Path(DataPaths.TRAINED_MODEL_PATH).name
        building_prep_path = output_dir / Path(DataPaths.BUILDING_PREPROCESSOR_PATH).name
        gm_prep_path = output_dir / Path(DataPaths.GM_PREPROCESSOR_PATH).name
        response_scaler_path = output_dir / Path(DataPaths.RESPONSE_SCALER_PATH).name

        import torch

        torch.save(trainer.model.state_dict(), model_path)
        building_preprocessor.save(str(building_prep_path))
        gm_preprocessor.save(str(gm_prep_path))
        response_scaler.save(str(response_scaler_path))

        logger.info('Saved model to %s', model_path)
        logger.info('Saved building preprocessor to %s', building_prep_path)
        logger.info('Saved GM preprocessor to %s', gm_prep_path)
        logger.info('Saved response scaler to %s', response_scaler_path)
        logger.info('Training complete')
    except Exception as exc:
        logger.error('Training failed: %s', exc)
        sys.exit(1)
