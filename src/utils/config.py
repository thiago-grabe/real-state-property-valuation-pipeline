from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    experiment_name: str
    model_name: str
    metric: str
    metric_goal: str = "min"
