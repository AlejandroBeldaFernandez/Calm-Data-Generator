from typing import Optional, Dict, List, Union, Any, Literal
from pydantic import BaseModel, Field, ConfigDict


class DateConfig(BaseModel):
    """
    Configuration for timestamp injection in generated data.
    """

    date_col: str = "timestamp"
    start_date: Optional[str] = None
    frequency: int = 1
    step: Optional[Dict[str, int]] = None  # e.g. {"days": 1}


class DriftConfig(BaseModel):
    """
    Configuration for drift injection.
    """

    method: str = "inject_feature_drift"
    drift_type: str = "gaussian_noise"
    feature_cols: Optional[List[str]] = None
    magnitude: float = 0.2

    # Selection parameters
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    block_index: Optional[int] = None
    block_column: Optional[str] = None
    time_start: Optional[str] = None
    time_end: Optional[str] = None

    # Advanced / Window parameters
    center: Optional[Union[int, float]] = None
    width: Optional[Union[int, float]] = None
    profile: str = "sigmoid"
    speed_k: float = 1.0
    direction: str = "up"
    inconsistency: float = 0.0

    # For specialized drifts
    drift_value: Optional[float] = None
    drift_values: Optional[Dict[str, float]] = None

    # Extra params for specific methods (e.g., custom params for custom drift)
    params: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")  # Allow extra fields for flexibility


class EvolutionFeatureConfig(BaseModel):
    """
    Configuration for evolving a single feature.
    """

    type: str  # 'linear', 'cycle', 'sigmoid'
    slope: Optional[float] = 0.0
    intercept: Optional[float] = 0.0
    amplitude: Optional[float] = 1.0
    period: Optional[float] = 100.0
    phase: Optional[float] = 0.0
    center: Optional[float] = None
    width: Optional[float] = None
    
    # Extra parameters for specific evolution types
    rate: Optional[float] = None      # for exponential_growth, decay
    noise_std: Optional[float] = None # for noise
    step_std: Optional[float] = None  # for random_walk
    step: Optional[float] = None      # for step
    value: Optional[float] = None     # for step

    # Fields for evolve_type "driven_by" (Pilar 5)
    driver_col: Optional[str] = None      # column whose current value drives the delta
    func: Optional[str] = None            # "linear"|"exponential"|"power"|"polynomial"
    func_params: Optional[dict] = None    # parameters for func

    model_config = ConfigDict(extra="allow")  # Allow extra fields for flexibility


class ScenarioConfig(BaseModel):
    """
    Configuration for scenario injection (evolution and target construction).
    """

    state_config: Optional[Dict] = None
    evolve_features: Dict[str, Union[Dict, EvolutionFeatureConfig]] = Field(
        default_factory=dict
    )
    construct_target: Optional[Dict] = None


class ReportConfig(BaseModel):
    """
    Configuration for report generation.
    """

    output_dir: str = "output"
    auto_report: bool = True
    minimal: bool = False
    target_column: Optional[str] = None
    time_col: Optional[str] = None
    block_column: Optional[str] = None
    resample_rule: Optional[Union[str, int]] = None
    privacy_check: bool = False
    adversarial_validation: bool = False
    focus_columns: Optional[List[str]] = None
    constraints_stats: Optional[Dict[str, int]] = None
    sequence_config: Optional[Dict] = None
    per_block_external_reports: bool = False
