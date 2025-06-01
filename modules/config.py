"""
Configuration classes and validation functions.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Literal


def sanitize_config_for_logging(config: "BenchmarkConfig") -> Dict[str, Any]:
    """Create a sanitized version of config for logging without sensitive data"""
    config_dict = asdict(config)
    # Replace sensitive fields with masked values
    sensitive_fields = ['api_key']
    for field in sensitive_fields:
        if field in config_dict and config_dict[field]:
            # Show only first 4 and last 4 characters if longer than 8, otherwise mask completely
            value = config_dict[field]
            if len(value) > 8:
                config_dict[field] = f"{value[:4]}...{value[-4:]}"
            else:
                config_dict[field] = "***masked***"
    return config_dict


def sanitize_config_for_saving(config: "BenchmarkConfig") -> Dict[str, Any]:
    """Create a sanitized version of config for saving to files without sensitive data"""
    config_dict = asdict(config)
    # Remove sensitive fields entirely from saved data
    sensitive_fields = ['api_key']
    for field in sensitive_fields:
        if field in config_dict:
            config_dict[field] = "***removed_for_security***"
    return config_dict


def validate_config(config: "BenchmarkConfig") -> List[str]:
    """Validate configuration and return list of issues"""
    issues = []
    
    if config.max_complexity < 1 or config.max_complexity > 5:
        issues.append("max_complexity must be between 1 and 5")
    
    if config.trials_per_complexity < 1:
        issues.append("trials_per_complexity must be at least 1")
    
    if not config.content_types:
        issues.append("content_types cannot be empty")
    
    if config.temperature < 0.0 or config.temperature > 2.0:
        issues.append("temperature must be between 0.0 and 2.0")
    
    if config.creative_temperature < 0.0 or config.creative_temperature > 2.0:
        issues.append("creative_temperature must be between 0.0 and 2.0")
    
    if config.verification_temperature < 0.0 or config.verification_temperature > 2.0:
        issues.append("verification_temperature must be between 0.0 and 2.0")
    
    if config.transform_temperature < 0.0 or config.transform_temperature > 2.0:
        issues.append("transform_temperature must be between 0.0 and 2.0")
    
    if config.max_retries < 1:
        issues.append("max_retries must be at least 1")
    
    if config.verification_attempts < 1:
        issues.append("verification_attempts must be at least 1")
    
    if config.verification_aggregation not in ['best', 'avg', 'worst']:
        issues.append("verification_aggregation must be 'best', 'avg', or 'worst'")
    
    # Validate content types
    valid_content_types = {"code", "text", "data", "configuration", "documentation"}
    invalid_types = set(config.content_types) - valid_content_types
    if invalid_types:
        issues.append(f"Invalid content types: {', '.join(invalid_types)}. Valid types: {', '.join(valid_content_types)}")
    
    return issues


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark run"""
    # Default base URLs
    base_url: str = "http://localhost:1234/v1"
    creative_base_url: str = ""  # If empty, uses base_url
    verification_base_url: str = ""  # If empty, uses base_url
    transform_base_url: str = ""  # If empty, uses base_url
    
    # API configuration
    api_key: str = "not-needed"
    temperature: float = 0.3
    
    # Individual temperature settings
    creative_temperature: float = 0.7
    verification_temperature: float = 0.1
    transform_temperature: float = 0.3
    
    # Model configuration
    model_name: str = ""
    creative_model: str = ""
    verification_model: str = ""
    transform_model: str = ""
    
    # Benchmark configuration
    max_complexity: int = 5
    trials_per_complexity: int = 3
    content_types: List[str] = None
    topic: Optional[str] = None
    
    # Retry configuration
    max_retries: int = 3
    
    # Verification configuration
    verification_attempts: int = 1
    verification_aggregation: Literal['best', 'avg', 'worst'] = 'avg'
    
    # Logging configuration
    log_file: str = None
    
    def __post_init__(self):
        if self.content_types is None:
            self.content_types = ["code", "text", "data", "configuration", "documentation"]
        
        # Set default base URLs if not specified
        if not self.creative_base_url:
            self.creative_base_url = self.base_url
        if not self.verification_base_url:
            self.verification_base_url = self.base_url
        if not self.transform_base_url:
            self.transform_base_url = self.base_url


@dataclass
class BenchmarkResult:
    """Results from a single benchmark trial"""
    complexity_level: int
    content_type: str
    generated_content: str
    transformation_instruction: str
    transformed_content: str
    verification_score: float
    instruction_completion_rate: float
    execution_time: float
    errors: List[str]
    content_variations: List[str]
    specific_scores: Dict[str, float]
    
    # New retry-related fields
    content_generation_attempts: int = 1
    instruction_generation_attempts: int = 1
    transformation_attempts: int = 1
    verification_attempts: int = 1
    verification_scores: List[float] = None  # All verification scores when multiple attempts
    
    def __post_init__(self):
        if self.verification_scores is None:
            self.verification_scores = [self.verification_score]


@dataclass
class BenchmarkSummary:
    """Summary of all benchmark results"""
    total_trials: int
    average_score: float
    average_completion_rate: float
    scores_by_complexity: Dict[int, float]
    completion_rates_by_complexity: Dict[int, float]
    error_rate: float
    total_time: float
    
    # New retry-related summary fields
    average_content_attempts: float = 1.0
    average_instruction_attempts: float = 1.0
    average_transformation_attempts: float = 1.0
    average_verification_attempts: float = 1.0