"""Configuration and constants for the tall building surrogate model.

Centralizes feature definitions (10 building + 12 ground motion), all 100 response
variables, and hyperparameters for the MIMO neural network and optimization.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

# Building design features (X1 to X10)
BUILDING_FEATURES = [
    'Top plan geometry side count (X1)',
    'Bottom plan geometry side count (X2)',
    'Top and Bottom plan orientation (X3)',
    'Number of stories (X4)',
    'Floor height (X5)',
    'Vertical transformation method (X6)',
    'Tapering (X7)',
    'Twisting Angle (X8)',
    'Curvilinear; location of control floor (X9)',
    'Curvilinear; scale of control floor (X10)',
]

# Ground motion seismic parameters
GROUND_MOTION_FEATURES = [
    ' Mean Squared Error',
    ' Scale Factor',
    ' 5-75% Duration (sec)',
    ' 5-95% Duration (sec)',
    ' Arias Intensity (m/sec)',
    ' Magnitude',
    ' Mechanism',
    ' Rjb (km)',
    ' Rrup (km)',
    ' Vs30 (m/sec)',
    ' Lowest Useable Frequency (Hz)',
    ' Initial-search Scale Factor',
]

# All response variables the model predicts (100 total)
ALL_RESPONSE_COLUMNS = [
    " Magnitude",
    " Mechanism",
    " Rjb (km)",
    " Rrup (km)",
    " Vs30 (m/sec)",
    " Lowest Useable Frequency (Hz)",
    " Initial-search Scale Factor",
    "Overall_Max_Acc",
    "Acc_X",
    "Acc_Y",
    "Acc_Time",
    "Acc_Floor",
    "Max_Displacement",
    "Displacement_X",
    "Displacement_Y",
    "Disp_Time",
    "Disp_Story",
    "Overall_Max_Drift",
    "Drift_X",
    "Drift_Y",
    "Drift_Time",
    "Drift_Floor",
    "Max_Stress_Tension_diagrid",
    "Stress_Tens_Time_diagrid",
    "Stress_Tens_Story_diagrid",
    "Max_Stress_Compression_diagrid",
    "Stress_Comp_Time_diagrid",
    "Stress_Comp_Story_diagrid",
    "Max_Shear_diagrid",
    "Shear_Time_diagrid",
    "Shear_Story_diagrid",
    "Max_Von_Mises_diagrid",
    "VM_Time_diagrid",
    "VM_Story_diagrid",
    "Total_Max_Stress_Tension_tot",
    "Total_Stress_Tens_Time_tot",
    "Total_Stress_Tens_Story_tot",
    "Total_Max_Stress_Compression_tot",
    "Total_Stress_Comp_Time_tot",
    "Total_Stress_Comp_Story_tot",
    "Total_Max_Shear_tot",
    "Total_Shear_Time_tot",
    "Total_Shear_Story_tot",
    "Total_Max_Von_Mises_tot",
    "Total_VM_Time_tot",
    "Total_VM_Story_tot",
    "Overall_Max_Torsion",
    "Torsion_Time",
    "Torsion_Floor",
    "Diagrid_Max_Magnitude_R",
    "Diagrid_Ry_Value",
    "Diagrid_Rz_Value",
    "Diagrid_Max_Time_R",
    "Diagrid_Max_Magnitude_M",
    "Diagrid_My_Value",
    "Diagrid_Mz_Value",
    "Diagrid_Max_Time_M",
    "Core_Max_Magnitude_R",
    "Core_Ry_Value",
    "Core_Rz_Value",
    "Core_Max_Time_R",
    "Core_Max_Magnitude_M",
    "Core_My_Value",
    "Core_Mz_Value",
    "Core_Max_Time_M",
    "Total_Max_Magnitude_R",
    "Total_Ry_Value",
    "Total_Rz_Value",
    "Total_Diagrid_R_Contrib_Value",
    "Total_Core_R_Contrib_Value",
    "Total_Max_Time_R",
    "Total_Max_Magnitude_M",
    "Total_My_Value",
    "Total_Mz_Value",
    "Total_Diagrid_M_Contrib_Value",
    "Total_Core_M_Contrib_Value",
    "Total_Max_Time_M",
    "Cost of structure",
    "Cost of floor",
    "Cost of Land",
    "Total costs",
    "Total costs/TGA",
    "selleble price",
    "EC Steel1",
    "EC Steel2",
    "EC Steel3",
    "EC floor1",
    "EC floor2",
    "EC total steel1, floor1",
    "EC total steel2, floor1",
    "EC total steel3, floor1",
    "EC total steel1, floor2",
    "EC total steel2, floor2",
    "EC total steel3, floor2",
    "EC GIA (Steel1, floor 1)",
    "EC GIA (Steel2, floor 1)",
    "EC GIA (Steel3, floor 1)",
    "EC GIA (Steel1, floor 2)",
    "EC GIA (Steel2, floor 2)",
    "EC GIA (Steel3, floor 2)",
]

# Key responses selected for design optimization based on structural engineering importance
PRIMARY_OPTIMIZATION_TARGETS = [
    'Overall_Max_Acc',          # Peak acceleration
    'Max_Displacement',          # Maximum lateral displacement
    'Overall_Max_Drift',         # Story drift ratio
    'Total_Max_Von_Mises_tot',   # Stress magnitude
    'Overall_Max_Torsion',       # Torsional response
    'Total_Max_Magnitude_R',     # Reaction moment magnitude
    'Total_Max_Magnitude_M',     # Moment magnitude
    'Total costs/TGA',           # Cost per floor area
]

# Response grouping for neural network output layers (one output per response)
RESPONSE_GROUPS: List[Tuple[int, int]] = [
    (i, i + 1) for i in range(len(ALL_RESPONSE_COLUMNS))
]


# ============================================================================
# NEURAL NETWORK ARCHITECTURE
# ============================================================================

@dataclass
class NeuralNetworkConfig:
    """Configuration for MIMO-FNN neural network architecture."""
    
    # Input branch sizes
    building_input_size: int = len(BUILDING_FEATURES)
    ground_motion_input_size: int = len(GROUND_MOTION_FEATURES)
    
    # Hidden layer sizes
    input_branch_hidden_size: int = 512
    shared_layer_1_size: int = 512
    shared_layer_2_size: int = 256
    shared_layer_3_size: int = 128
    
    # Regularization
    dropout_rate_1: float = 0.3
    dropout_rate_2: float = 0.3
    dropout_rate_3: float = 0.2
    l2_weight_decay: float = 1e-5
    
    # Training parameters
    learning_rate: float = 2e-5
    batch_size: int = 32
    num_epochs: int = 200
    k_fold_splits: int = 2
    test_size: float = 0.2
    random_state: int = 42
    
    # Device (cuda/cpu)
    device: str = 'cpu'  # Set to 'cuda' if GPU available

    @property
    def output_sizes(self) -> List[int]:
        """Return output head sizes derived from response groups."""
        return [end - start for start, end in RESPONSE_GROUPS]


# ============================================================================
# OPTIMIZATION PARAMETERS
# ============================================================================

@dataclass
class NSGAIIConfig:
    """Configuration for NSGA-II multi-objective optimization."""
    
    # Population parameters
    population_size: int = 100  # Size of population per generation
    num_generations: int = 10   # Number of generations to evolve
    
    # Crossover and mutation
    crossover_prob: float = 0.9   # Probability of crossover
    mutation_prob: float = 0.1    # Probability of mutation
    
    # Objectives for optimization (user-specified; default to structural priorities)
    objectives: List[str] = field(default_factory=lambda: PRIMARY_OPTIMIZATION_TARGETS)
    
    def __post_init__(self):
        if not self.objectives:
            self.objectives = PRIMARY_OPTIMIZATION_TARGETS


# ============================================================================
# DATA PATHS AND FILENAMES
# ============================================================================

class DataPaths:
    """Centralized data path definitions for the project."""
    
    # Input data
    DATABASE_CSV: str = 'data/database.csv'
    
    # Model outputs
    TRAINED_MODEL_PATH: str = 'models/mimo_fnn_model.pth'
    BUILDING_PREPROCESSOR_PATH: str = 'models/building_feature_preprocessor.pkl'
    GM_PREPROCESSOR_PATH: str = 'models/gm_feature_preprocessor.pkl'
    RESPONSE_SCALER_PATH: str = 'models/response_scaler.pkl'
    
    # Optimization outputs
    OPTIMIZED_SOLUTIONS_PATH: str = 'results/optimized_solutions.csv'
    TRAINING_HISTORY_PATH: str = 'results/training_history.csv'
    PERFORMANCE_METRICS_PATH: str = 'results/performance_metrics.json'


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_FORMAT: str = (
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

LOG_DATE_FORMAT: str = '%Y-%m-%d %H:%M:%S'

LOG_LEVEL: str = 'INFO'
