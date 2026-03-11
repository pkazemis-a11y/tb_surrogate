from pathlib import Path

import torch

import src

from src.core import (
    ALL_RESPONSE_COLUMNS,
    BUILDING_FEATURES,
    GROUND_MOTION_FEATURES,
    DataLoader,
    NeuralNetworkConfig,
)
from src.models import create_model


def test_package_imports_expose_reference_modeling_modules() -> None:
    """Verify the public package imports expose the cleaned surrogate modules."""
    assert hasattr(src, 'core')
    assert hasattr(src, 'models')
    assert hasattr(src, 'optimization')


def test_response_contract_matches_reference_config() -> None:
    """Verify the documented response contract matches the reference config."""
    config = NeuralNetworkConfig()

    assert sum(config.output_sizes) == len(ALL_RESPONSE_COLUMNS)
    assert len(config.output_sizes) == len(ALL_RESPONSE_COLUMNS)


def test_reference_model_output_matches_response_count() -> None:
    """Verify the reference model produces one scalar output per response."""
    config = NeuralNetworkConfig()
    model = create_model(config=config, output_sizes=config.output_sizes, device='cpu')

    building_batch = torch.zeros(4, len(BUILDING_FEATURES), dtype=torch.float32)
    gm_batch = torch.zeros(4, len(GROUND_MOTION_FEATURES), dtype=torch.float32)

    outputs = model(building_batch, gm_batch)

    assert len(outputs) == len(ALL_RESPONSE_COLUMNS)
    assert sum(output.shape[1] for output in outputs) == len(ALL_RESPONSE_COLUMNS)


def test_data_loader_extracts_all_responses() -> None:
    """Verify DataLoader extracts all 100 responses by default."""
    loader = DataLoader(str(Path('data') / 'database.csv'))

    data = loader.load()
    x_building, x_gm, y = loader.get_features_and_responses()

    assert not data.empty
    assert list(x_building.columns) == BUILDING_FEATURES
    assert list(x_gm.columns) == GROUND_MOTION_FEATURES
    assert list(y.columns) == ALL_RESPONSE_COLUMNS
    assert x_building.shape[1] == 10
    assert x_gm.shape[1] == 12
    assert y.shape[1] == 100
