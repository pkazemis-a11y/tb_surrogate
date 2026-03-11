"""Data loading and preprocessing module.

Loads CSV data, handles feature validation, scaling, and normalization.
Provides sklearn-like fit/transform interface for preprocessing pipelines.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.preprocessing import LabelEncoder
import joblib

from .config import (
    BUILDING_FEATURES, 
    GROUND_MOTION_FEATURES, 
    ALL_RESPONSE_COLUMNS,
    PRIMARY_OPTIMIZATION_TARGETS,
)


logger = logging.getLogger(__name__)


class DataLoader:
    """Loads and validates building design dataset from CSV."""
    
    def __init__(self, data_path: str):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to the CSV file containing the dataset.
            
        Raises:
            FileNotFoundError: If data file doesn't exist.
        """
        if not data_path.endswith('.csv'):
            raise ValueError("Data path must point to a CSV file")
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        self.data_path = data_path
        self.data: Optional[pd.DataFrame] = None
        logger.info(f"DataLoader initialized for: {data_path}")
    
    def load(self) -> pd.DataFrame:
        """
        Load dataset from CSV file.
        
        Returns:
            Loaded DataFrame.
            
        Raises:
            FileNotFoundError: If file cannot be found.
            pd.errors.ParserError: If CSV is malformed.
        """
        try:
            self.data = pd.read_csv(self.data_path)
            logger.info(f"Loaded data with shape: {self.data.shape}")
            logger.info(f"Columns: {list(self.data.columns)}")
            return self.data
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.data_path}")
            raise
        except pd.errors.ParserError as e:
            logger.error(f"Error parsing CSV file: {e}")
            raise
    
    def get_features_and_responses(
        self,
        building_features: List[str] = BUILDING_FEATURES,
        ground_motion_features: List[str] = GROUND_MOTION_FEATURES,
        responses: List[str] = ALL_RESPONSE_COLUMNS,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Extract building features, ground motion features, and response variables.
        
        By default, extracts all 91 response columns. Can be overridden to extract
        a subset (e.g., for optimization-focused training).
        
        Args:
            building_features: Building design feature column names.
            ground_motion_features: Seismic parameter column names.
            responses: Response variable column names (default: all 91).
            
        Returns:
            Tuple of (X_building, X_gm, y)
            
        Raises:
            ValueError: If required columns are missing.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")
        
        # Validate columns exist
        all_required = (
            building_features + 
            ground_motion_features + 
            responses
        )
        
        missing_cols = [col for col in all_required if col not in self.data.columns]
        if missing_cols:
            raise ValueError(
                f"Missing columns in dataset: {missing_cols}"
            )
        
        X_building = self.data[building_features].copy()
        X_gm = self.data[ground_motion_features].copy()
        y = self.data[responses].copy()
        
        logger.info(f"Extracted {X_building.shape[1]} building features")
        logger.info(f"Extracted {X_gm.shape[1]} ground motion features")
        logger.info(f"Extracted {y.shape[1]} response variables")
        
        return X_building, X_gm, y


class FeaturePreprocessor:
    """
    Preprocesses features through scaling, encoding, and feature engineering.
    
    Follows sklearn's fit/transform pattern for consistency.
    """
    
    def __init__(self, categorical_features: Optional[List[str]] = None):
        """
        Initialize preprocessor.
        
        Args:
            categorical_features: List of categorical feature names to encode.
        """
        self.categorical_features = categorical_features or []
        self.scaler = StandardScaler()
        self.label_encoders: dict = {}
        self.fill_values: Dict[str, object] = {}
        self.is_fitted = False

    def _prepare_features(self, X: pd.DataFrame, fit_mode: bool) -> pd.DataFrame:
        """Apply imputations and categorical encoding before scaling."""
        X_copy = X.copy()

        for col in X_copy.columns:
            if col in self.categorical_features:
                if fit_mode:
                    fill_value = '__missing__'
                    self.fill_values[col] = fill_value
                fill_value = self.fill_values.get(col, '__missing__')
                X_copy[col] = X_copy[col].fillna(fill_value).astype(str)
            else:
                if fit_mode:
                    fill_value = float(X_copy[col].median()) if not X_copy[col].dropna().empty else 0.0
                    self.fill_values[col] = fill_value
                fill_value = self.fill_values.get(col, 0.0)
                X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce').fillna(fill_value)

        for col in self.categorical_features:
            if col in X_copy.columns:
                if fit_mode:
                    le = LabelEncoder()
                    le.fit(X_copy[col])
                    self.label_encoders[col] = le
                unknown_values = set(X_copy[col].unique()) - set(self.label_encoders[col].classes_)
                if unknown_values:
                    raise ValueError(
                        f"Unknown categorical values for column '{col}': {sorted(unknown_values)}"
                    )
                X_copy[col] = self.label_encoders[col].transform(X_copy[col])

        return X_copy
        
    def fit(self, X: pd.DataFrame) -> 'FeaturePreprocessor':
        """
        Fit preprocessor on training data.
        
        Args:
            X: Input features DataFrame.
            
        Returns:
            Self for method chaining.
        """
        prepared = self._prepare_features(X, fit_mode=True)
        self.scaler.fit(prepared)
        
        self.is_fitted = True
        logger.info("Feature preprocessor fitted")
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted preprocessor.
        
        Args:
            X: Input features DataFrame.
            
        Returns:
            Transformed feature array (samples x features).
            
        Raises:
            ValueError: If preprocessor not fitted before transform.
        """
        if not self.is_fitted:
            raise ValueError(
                "Preprocessor must be fitted before transform. Call fit() first."
            )
        
        prepared = self._prepare_features(X, fit_mode=False)
        X_scaled = self.scaler.transform(prepared)
        
        return X_scaled
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform features in one step.
        
        Args:
            X: Input features DataFrame.
            
        Returns:
            Transformed feature array.
        """
        return self.fit(X).transform(X)
    
    def save(self, path: str):
        """Save fitted preprocessor to disk."""
        joblib.dump(self, path)
        logger.info(f"Preprocessor saved to {path}")
    
    @staticmethod
    def load(path: str) -> 'FeaturePreprocessor':
        """Load fitted preprocessor from disk."""
        preprocessor = joblib.load(path)
        logger.info(f"Preprocessor loaded from {path}")
        return preprocessor


class ResponseScaler:
    """
    Handles scaling of response variables independently from features.
    
    Uses MaxAbsScaler to preserve sign information (important for some responses).
    """
    
    def __init__(self, scaler_type: str = 'maxabs'):
        """
        Initialize response scaler.
        
        Args:
            scaler_type: Type of scaler ('maxabs', 'standard', or 'minmax').
        """
        if scaler_type == 'maxabs':
            self.scaler = MaxAbsScaler()
        elif scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        self.is_fitted = False
        self.scaler_type = scaler_type
    
    def fit(self, y: np.ndarray) -> 'ResponseScaler':
        """
        Fit scaler on response data.
        
        Args:
            y: Response values (samples x responses).
            
        Returns:
            Self for method chaining.
        """
        self.scaler.fit(y)
        self.is_fitted = True
        logger.info(f"Response scaler ({self.scaler_type}) fitted")
        return self
    
    def transform(self, y: np.ndarray) -> np.ndarray:
        """Transform response values."""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
        return self.scaler.transform(y)
    
    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        """Scale responses back to original space."""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse_transform")
        return self.scaler.inverse_transform(y_scaled)
    
    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(y).transform(y)
    
    def save(self, path: str):
        """Save fitted scaler to disk."""
        joblib.dump(self, path)
        logger.info(f"Response scaler saved to {path}")
    
    @staticmethod
    def load(path: str) -> 'ResponseScaler':
        """Load fitted scaler from disk."""
        scaler = joblib.load(path)
        logger.info(f"Response scaler loaded from {path}")
        return scaler
