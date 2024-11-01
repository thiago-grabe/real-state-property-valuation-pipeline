from dataclasses import dataclass

from typing import Literal

METRICS = Literal["mse", "mae", "mape"]
METRIC_GOAL = Literal["min", "max"]

@dataclass
class ExperimentConfig:
    experiment_name: str
    model_name: str
    metric: METRICS
    metric_goal: METRIC_GOAL = "min"
