"""Example: Load and explore the tall building design database.

Demonstrates data loading, feature extraction, and basic statistics
using the project's DataLoader utility with built-in validation.
"""

import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core import (
    ALL_RESPONSE_COLUMNS,
    BUILDING_FEATURES,
    DataLoader,
    GROUND_MOTION_FEATURES,
    PRIMARY_OPTIMIZATION_TARGETS,
)


def main():
    """Load dataset and display key information."""
    # Load and validate data
    loader = DataLoader('data/database.csv')
    data = loader.load()
    
    print("Dataset Overview")
    print("=" * 70)
    print(f"Total samples: {data.shape[0]:,}")
    print(f"Total columns: {data.shape[1]}")
    
    # Extract features and all responses
    X_building, X_gm, y = loader.get_features_and_responses()
    
    print("\nFeature Inputs")
    print("=" * 70)
    print(f"Building design features: {X_building.shape[1]} variables")
    for feat in BUILDING_FEATURES:
        print(f"  • {feat}")
    
    print(f"\nGround motion parameters: {X_gm.shape[1]} variables")
    for feat in GROUND_MOTION_FEATURES:
        print(f"  • {feat}")
    
    print("\nResponse Variables")
    print("=" * 70)
    print(f"Model predicts: {y.shape[1]} total response variables")
    print(f"Optimization targets: {len(PRIMARY_OPTIMIZATION_TARGETS)} key responses")
    for target in PRIMARY_OPTIMIZATION_TARGETS:
        print(f"  ✓ {target}")
    
    # Response range statistics
    print("\n" + "=" * 70)
    print("=" * 70)
    print(y.describe())
    
    # Correlations (example of analysis)
    print("\n" + "=" * 70)
    print("Feature-Response Correlations (sample)")
    print("=" * 70)
    
    # Show correlation of first building feature with responses
    first_feature = BUILDING_FEATURES[0]
    if first_feature in data.columns:
        corr = pd.concat([X_building[[first_feature]], y], axis=1).corr().iloc[0, 1:]
        print(f"\n{first_feature} correlations with responses:")
        print(corr.sort_values(ascending=False).head(10))


if __name__ == '__main__':
    main()
