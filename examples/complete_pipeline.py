"""Example training workflow using the package API.

The script loads the dataset, fits preprocessors, trains the surrogate model,
and saves the resulting artifacts to the models directory.
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core import (
    DataLoader,
    FeaturePreprocessor,
    ResponseScaler,
    NeuralNetworkConfig,
)
from src.models import ModelTrainer, create_model

import torch


def setup_logging():
    """Configure basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Run the training example."""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info('Loading data')

    loader = DataLoader('data/database.csv')
    _ = loader.load()
    X_building, X_gm, y = loader.get_features_and_responses()

    building_prep = FeaturePreprocessor()
    gm_prep = FeaturePreprocessor()
    X_building_scaled = building_prep.fit_transform(X_building)
    X_gm_scaled = gm_prep.fit_transform(X_gm)

    response_scaler = ResponseScaler(scaler_type='maxabs')
    y_scaled = response_scaler.fit_transform(y.values)

    logger.info('Building features shape: %s', X_building_scaled.shape)
    logger.info('Ground-motion features shape: %s', X_gm_scaled.shape)
    logger.info('Response matrix shape: %s', y_scaled.shape)

    logger.info('Training model')

    config = NeuralNetworkConfig(
        learning_rate=2e-5,
        batch_size=32,
        num_epochs=200,
        k_fold_splits=2,
    )
    
    model = create_model(
        config=config,
        output_sizes=config.output_sizes,
        device='cpu',
    )

    trainer = ModelTrainer(config, model, device='cpu')
    trainer.train_final_model(X_building_scaled, X_gm_scaled, y_scaled)

    Path('models').mkdir(exist_ok=True)
    torch.save(model.state_dict(), 'models/mimo_fnn_model.pth')
    building_prep.save('models/building_feature_preprocessor.pkl')
    gm_prep.save('models/gm_feature_preprocessor.pkl')
    response_scaler.save('models/response_scaler.pkl')

    logger.info('Artifacts saved to models/')
    logger.info('Run scripts/optimize_designs.py or the optimize-designs CLI to generate candidate designs.')


if __name__ == '__main__':
    main()
