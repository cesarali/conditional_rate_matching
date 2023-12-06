from dataclasses import dataclass

@dataclass
class RealFlipConfig:
    name: str = "RealFlip"

@dataclass
class GradientEstimatorConfig:
    name: str = "GradientEstimator"
    stein_epsilon: float = 1e-3
    stein_sample_size: int = 150

@dataclass
class SteinSpinEstimatorConfig:
    name : str = "SteinSpinEstimator"
    stein_epsilon :float = 0.2
    stein_sample_size :int = 200


all_flip_configs = {"GradientEstimator":GradientEstimatorConfig,
                    "SteinSpinEstimator":SteinSpinEstimatorConfig,
                    "RealFlipConfig":RealFlipConfig}