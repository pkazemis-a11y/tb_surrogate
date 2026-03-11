from pathlib import Path

from src.core import (
    ALL_RESPONSE_COLUMNS,
    BUILDING_FEATURES,
    GROUND_MOTION_FEATURES,
    DataLoader,
    NeuralNetworkConfig,
)


def test_network_output_matches_response_count() -> None:
    """Verify neural network output size matches all response variables."""
    config = NeuralNetworkConfig()
    
    # Model should have one output per response (100 total)
    assert sum(config.output_sizes) == len(ALL_RESPONSE_COLUMNS)
    assert len(config.output_sizes) == len(ALL_RESPONSE_COLUMNS)


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
    assert y.shape[1] == 100  # All response columns